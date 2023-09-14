"""
Tools to decode single stages/layers
"""

import logging
import struct
from functools import reduce
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
from constants import ChannelIndices, OpModes, OpType, PadModes, PoolType, Stage

import export_to_xml as XML
from inout_decoder import InOutDecoder
from export_to_xml import Layer as XLayer, XMLExporter
from export_to_xml import Edge as XEdge
from export_to_bin import BinExporter
from print_helper import PrintHelper

class DataReader:
    """
    Helper class to read and interpret data in various ways.
    """
    def __init__(self, buffer):
        self.buffer = buffer
        self.ptr = 0

    def reset_buf(self, buffer = None, pos = 0):
        """
        Reset buffer to `buffer` (default None) and set position to `pos` (default 0).
        """
        if buffer:
            self.buffer = buffer
        self.ptr = pos

    def skip(self, num):
        """
        Skip `num` bytes of buffer.
        """
        self.ptr += num

    def read_float(self, count = 1):
        """
        Read 32-bit float from buffer.
        """
        if count > 1:
            return self._read(f'{count}f', 4*count)
        return self._read('f', 4)

    def read_uint(self, count = 1):
        """
        Read 32-bit unsigned integer from buffer.
        """
        if count > 1:
            return self._read(f'{count}I', 4*count)
        return self._read('I', 4)

    def read_int(self, count = 1):
        """
        Read 32-bit signed integer from buffer.
        """
        if count > 1:
            return self._read(f'{count}i', 4*count)
        return self._read('i', 4)

    def _read(self, code, typelen):
        val = struct.unpack(code, self.buffer[self.ptr:self.ptr+typelen])
        self.ptr += typelen
        if len(val) == 1:
            return val[0]
        return val


class StageDecoder:
    """
    Decodes several different stages.
    """

    def __init__(self, printer: PrintHelper, out_bin, out_xml, cbuf, layer_names, xml_exporter: XMLExporter, mnist: bool):
        self.current_shape = (-1, -1, -1, -1)
        self.bin_exporter = BinExporter(filename=out_bin, printer=printer)
        self.xml_filename = out_xml
        self.layer_names = layer_names
        self.split_shapes = []
        self.printer = printer
        self.experimental_dense_detection = True # experimental feature!
        self.is_flattened = False # experimental feature!
        self.mnist_special_treatment = mnist

        # Change this if not using default MNIST model!
        self.scale = np.float16(0.0625)

        self.was_permuted = False
        self.last_permutation = [0, 1, 2, 3]
        self.last_permutation_index = -1
        self.printer = printer
        self.maybe_split = False

        self.data_buffer = cbuf.getbuffer()
        self.io_decoder = InOutDecoder(self.data_buffer, printer)

        self.xml_exporter = xml_exporter

        self.layer_count = {
            'reshape': 0,
            'convolution': 0,
            'pooling': 0,
            'transpose': 0,
            'add': 0,
            'matmul': 0
        }

    def __repr__(self):
        return 'StageDecoder'

    def finalize(self):
        """
        Finalize stage decoding by closing the binary output file.
        """
        self.bin_exporter.finalize()
        self.xml_exporter.write_to_file(filename=self.xml_filename)
    
    def _match_dimensions(self, new_shape):
        """
        Adds Reshape layers whenever needed.
        """

        self.printer.print(f"Current shape info: {self.current_shape}", loglevel=logging.INFO)
        self.printer.print(f"New shape info: {new_shape}", loglevel=logging.INFO)
        if self.current_shape == new_shape:
            self.printer.print("All good, dimensions match. No need to take action.", loglevel=logging.DEBUG)
            return
        if self.current_shape == (-1, -1, -1, -1) and self.xml_exporter.layer_id == 0: # no layer added yet -> this is our input node?!
            self.printer.print(f"Initially set shape to {new_shape}. All good.", loglevel=logging.INFO)
            input_layer = {
                'name': self.layer_names['input'],
                'type': 'Parameter',
                'version': 'opset1',
                'data': {
                    'shape': str.join(', ', [str(x) for x in new_shape]),
                    'element_type': 'f16'
                },
                'inputs': [],
                'out_prec': 'FP16',
                'outputs': [new_shape]
            }
            self.xml_exporter.add_layer(input_layer)
            self.current_shape = new_shape
            self.printer.print(f"changed current shape to {self.current_shape}", loglevel=logging.INFO)
            return
        
        self.printer.print(f"There is a mismatch: {self.current_shape} vs. {new_shape}", loglevel=logging.INFO)
        if len(self.current_shape) == len(new_shape):
            self.printer.print("We skip this for testing!", loglevel=logging.INFO)
        elif len(self.current_shape) != len(new_shape):
            if len(self.current_shape) < len(new_shape) and self.is_flattened:
                self.printer.print(f"We are in flat mode and this would increase from {self.current_shape} to {new_shape}. Skipping this one.", loglevel=logging.WARN)
                return
            self.printer.print(f"Shape size did not match, so we add a Reshape layer here from {self.current_shape} to {new_shape}", loglevel=logging.WARN)
            self._inject_reshape(new_shape)
            
    def decode(self, stage_data, in_out_desc:Tuple=None) -> Tuple[List, List]:
        """
        Wrapper around `decode_inouts` for convenience.
        """
        new_shape, new_data = self.io_decoder.decode_inouts(stage_data, in_out_desc=in_out_desc, inshape=self.current_shape)
        self._match_dimensions(new_shape[0])
        self.printer.print(new_shape, loglevel=logging.ERROR)
        return new_shape, new_data

    def _inject_reshape(self, dst_dimension):
        """
        Adds Reshape layer in case dimensions do not match up.
        """

        last_layer_id = self.xml_exporter.layer_id - 1
        last_layer_port = self.xml_exporter.layer_ports - 1

        reshape_params_layer = XML.add_const(
            name=f"reshape_{self.layer_count['reshape']}/Shape",
            data={
                'element_type': 'i64',
                'shape': str(len(dst_dimension)),
                'offset': str(self.bin_exporter.offset),
                'size': str(len(dst_dimension)*8)
            },
            shape=[(len(dst_dimension),)],
            prec='I64')
        reshape_const_layer_id, reshape_const_layer_ports = self.xml_exporter.add_layer(reshape_params_layer)

        reshape_layer = XML.add_reshape(
            name=f"reshape_{self.layer_count['reshape']}",
            inshape=self.current_shape,
            indexshape=(len(dst_dimension),),
            outshape=dst_dimension)
        reshape_layer_id, reshape_layer_ports = self.xml_exporter.add_layer(reshape_layer)

        input_to_reshape_edge = XEdge(
            XLayer(layer_id = last_layer_id, port_id = last_layer_port),
            XLayer(layer_id = reshape_layer_id, port_id = reshape_layer_ports['input'][0])
        )
        self.xml_exporter.add_edge(input_to_reshape_edge)
        const_to_reshape_edge = XEdge(
            XLayer(layer_id = reshape_const_layer_id, port_id = reshape_const_layer_ports['output'][0]),
            XLayer(layer_id = reshape_layer_id, port_id = reshape_layer_ports['input'][1])
        )
        self.xml_exporter.add_edge(const_to_reshape_edge)
        self.printer.print("Added layer and corresponding edges. Now updating data", loglevel=logging.INFO)

        bindata = np.int64(dst_dimension)
        if bindata[0] == 1:
            bindata[0] = -1 # seems like the model optimizer fills the first dimension (number of batches) with a -1
                            # which stands for 'undefined'/'unknown'?!
        self.bin_exporter.write_bytes(bytes(bindata))
        self.current_shape = dst_dimension

        self.layer_count['reshape'] += 1


    def decode_convert(self, data):
        """
        Decodes Convert stage.
        """
        reader = DataReader(data)
        scale, bias = reader.read_float(count = 2)
        convert_from_det_output = bool(reader.read_int())
        have_batch = bool(reader.read_int())
        self.printer.print(f"Identified Convert stage (scale: {scale}, bias: {bias}, convertFromDetOutput: {convert_from_det_output}, haveBatch: {have_batch})", loglevel=logging.INFO)
        self.printer.print("  (Default param values:  scale: 1.0, bias: 0.0, convertFromDetOutput: False, haveBatch: True)", loglevel=logging.DEBUG)
        stage_data = data[reader.ptr:]
        self.decode(stage_data, in_out_desc=('IN', 'OUT'))

    def decode_sum(self, data):
        """
        Decode Sum stage.
        """
        self.printer.print("Identified Sum stage (this is actually an Eltwise stage).", loglevel=logging.ERROR)
        # See openvino/inference-engine/src/vpu/graph_transformer/src/stages/eltwise.cpp:183-207
        reader = DataReader(data)
        self.printer.print(f" coeff1: {reader.read_float()}", loglevel=logging.DEBUG)
        self.printer.print(f" coeff2: {reader.read_float()}", loglevel=logging.DEBUG)
        post_op = Stage(reader.read_int())
        self.printer.print(f" postOperation: {post_op.name} ({hex(post_op)})", loglevel=logging.INFO)
        self.printer.print(f" negativeSlope: {reader.read_float()}", loglevel=logging.DEBUG)
        self.printer.print(f" min_value: {reader.read_float()}", loglevel=logging.DEBUG)
        self.printer.print(f" max_value: {reader.read_float()}", loglevel=logging.DEBUG)
        self.printer.print("Params End.")
        stage_data = data[reader.ptr:]
        self.decode(stage_data, in_out_desc=('IN0', 'OUT', 'IN1', 'IN2'))

    def decode_maxpool(self, data):
        """
        Decode MaxPool stage.
        """
        self.printer.print("Identified MaxPool stage (not implemented).", loglevel=logging.ERROR)
        self.decode(data)

    def decode_softmax(self, data):
        """
        Decode Softmax stage.
        """
        reader = DataReader(data)
        self.printer.print(f"Identified SoftMax stage (axisInd: {reader.read_uint()})", loglevel=logging.INFO)
        shapes, _ = self.decode(data[reader.ptr:], in_out_desc=('IN', 'OUT'))

        previous_id = self.xml_exporter.layer_id - 1
        previous_port = self.xml_exporter.layer_ports - 1
        softmax_layer = XML.add_softmax(self.layer_names['output'], shapes[0])
        softmax_id, softmax_port = self.xml_exporter.add_layer(softmax_layer)
        softmax_edge = XEdge(
            XLayer(layer_id = previous_id, port_id = previous_port),
            XLayer(layer_id = softmax_id, port_id = softmax_port['input'][0])
        )
        self.xml_exporter.add_edge(softmax_edge)

        result_sink = {
            'name': f"{self.layer_names['output']}/sink",
            'type': 'Result',
            'version': 'opset1',
            'data': {},
            'inputs': [shapes[0]],
            'outputs': [],
            'out_prec': 'FP16'
        }
        result_id, result_port = self.xml_exporter.add_layer(result_sink)
        softmax_result = XEdge(
            XLayer(layer_id = softmax_id, port_id = softmax_port['output'][0]),
            XLayer(layer_id = result_id, port_id = result_port['input'][0])
        )
        self.xml_exporter.add_edge(softmax_result)

 
    def decode_hwop(self, data):
        """
        Decodes MyriadXHwOp stage.
        """
        source_x = self.current_shape
        reader = DataReader(data)
        operations_ret = []
        vecsize, stype = reader.read_uint(count = 2)
        stype = OpType(stype)
        self.printer.print(f"Identified MyriadXHwOp stage (type: {stype.name}).", loglevel=logging.INFO)
        self.printer.print("  Params:", loglevel=logging.DEBUG)
        self.printer.print(f"    hwOps.vec.size = {vecsize}", loglevel=logging.INFO)
        first_iter_len = 0
        reader.reset_buf(pos = 4)
        pooltype = None
        for i in range(vecsize):
            if i > 0: # skip all following ops in this vector (they contain the same information)
                continue
            op_type = OpType(reader.read_uint())
            operations_ret.append(f'HW_{op_type.name}')
            self.printer.print(f"    hwOps.vec[{i}].opType = {op_type.name}", loglevel=logging.INFO)
            if op_type is OpType.POOL:
                pooltype = PoolType(reader.read_uint())
                self.printer.print(f"      hwOps.vec[{i}].poolType = {pooltype.name}", loglevel=logging.INFO)
            op_mode, with_pad = reader.read_uint(count = 2)
            with_pad = bool(with_pad)
            self.printer.print(f"    hwOps.vec[{i}].opMode = {OpModes(op_mode).name}", loglevel=logging.INFO)
            self.printer.print(f"    hwOps.vec[{i}].withPad = {with_pad}", loglevel=logging.DEBUG)
            if with_pad:
                self.printer.print(f"      hwOps.vec[{i}].padMode = {PadModes(reader.read_uint()).name}", loglevel=logging.ERROR)
            else:
                self.printer.print( f"      hwOps.vec[{i}].padMode = NONE(?)", loglevel=logging.ERROR)
            indices = reader.read_int(count = 5)
            self.printer.print(f"    hwOps.vec[{i}].inputInd  = {indices[ChannelIndices.INPUT]}", loglevel=logging.INFO)
            self.printer.print(f"    hwOps.vec[{i}].outputInd = {indices[ChannelIndices.OUTPUT]}", loglevel=logging.INFO)
            self.printer.print(f"    hwOps.vec[{i}].coeffsInd = {indices[ChannelIndices.COEFFS]}", loglevel=logging.INFO)
            self.printer.print(f"    hwOps.vec[{i}].biasesInd = {indices[ChannelIndices.BIASES]}", loglevel=logging.INFO)
            self.printer.print(f"    hwOps.vec[{i}].scalesInd = {indices[ChannelIndices.SCALES]}", loglevel=logging.INFO)
            outnum_chans = None
            kernel_width = None
            kernel_height = None
            if op_type is not OpType.FC:
                outchan_offset, outnum_chans = reader.read_uint(count = 2)
                self.printer.print(f"    hwOps.vec[{i}].outChanOffset = {outchan_offset}", loglevel=logging.DEBUG)
                self.printer.print(f"    hwOps.vec[{i}].outNumChans = {outnum_chans}", loglevel=logging.INFO)
            else:
                self.printer.print(f"    hwOps.vec[{i}].fcInputOffset = {reader.read_uint()}", loglevel=logging.DEBUG)
                self.printer.print(f"    hwOps.vec[{i}].fcInputNum = {reader.read_uint()}", loglevel=logging.DEBUG)
                self.printer.print(f"    hwOps.vec[{i}].fcOutputOffset = {reader.read_uint()}", loglevel=logging.DEBUG)
                self.printer.print(f"    hwOps.vec[{i}].fcOutputNum = {reader.read_uint()}", loglevel=logging.DEBUG)
                self.printer.print(f"    hwOps.vec[{i}].fcAccum = {reader.read_uint()}", loglevel=logging.DEBUG)
            if op_type is not OpType.FC:
                kernel_width, kernel_height, kernel_stride  = reader.read_uint(count = 3)
                self.printer.print(f"    hwOps.vec[{i}].kernelWidth = {kernel_width}", loglevel=logging.INFO)
                self.printer.print(f"    hwOps.vec[{i}].kernelHeight = {kernel_height}", loglevel=logging.INFO)
                self.printer.print(f"    hwOps.vec[{i}].kernelStride = {kernel_stride}", loglevel=logging.INFO)
            if op_type is OpType.CONVPOOL:
                self.printer.print(f"    hwOps.vec[{i}].poolKernelWidth = {reader.read_uint()}", loglevel=logging.DEBUG)
                self.printer.print(f"    hwOps.vec[{i}].poolKernelHeight = {reader.read_uint()}", loglevel=logging.DEBUG)
            with_relu = bool(reader.read_uint())
            self.printer.print(f"    hwOps.vec[{i}].withReLU = {with_relu}", loglevel=logging.DEBUG)
            if with_relu:
                t_0, a_0, a_1 = reader.read_uint(count = 3)
                self.printer.print(f"      hwOps.vec[{i}].t0 = {t_0}", loglevel=logging.DEBUG)
                self.printer.print(f"      hwOps.vec[{i}].a0 = {a_0}", loglevel=logging.DEBUG)
                self.printer.print(f"      hwOps.vec[{i}].a1 = {a_1}", loglevel=logging.DEBUG)
            with_clamp = bool(reader.read_uint())
            self.printer.print(f"    hwOps.vec[{i}].withClamp = {with_clamp}", loglevel=logging.DEBUG)

            if with_clamp:
                self.printer.print(f"      hwOps.vec[{i}].clampMaxVal = {reader.read_uint()}", loglevel=logging.DEBUG)
            self.printer.print(f"    hwOps.vec[{i}].reuseData = {reader.read_int()}", loglevel=logging.DEBUG)
            self.printer.print(f"    hwOps.vec[{i}].reuseCoeff = {reader.read_int()}", loglevel=logging.DEBUG)

            first_iter_len = reader.ptr - 4 # subtract 4 bytes (vecsize is not part of the first iteration)
        reader.skip(first_iter_len * (vecsize - 1)) # one iter_len has been read already ;)
        injected = bool(reader.read_uint())
        self.printer.print(f"    injectedStage: {injected}", loglevel=logging.WARN if injected else logging.DEBUG)
        if injected:
            self.printer.print("We need to deserialize the injected stage...", loglevel=logging.WARN)
            injected_stage_len, injected_stage_type = reader.read_uint(count = 2)
            injected_stage_type = Stage(injected_stage_type)
            self.printer.print(f"Stage len {injected_stage_len}", loglevel=logging.WARN)
            self.printer.print(f"Stage type {Stage(injected_stage_type).name} ({hex(injected_stage_type.value)})", loglevel=logging.WARN)

            self.printer.print("Pre-read inj-stage", loglevel=logging.ERROR)
            
            ########### COPIED FROM stage_decoder.py ##########
            mapper: Dict[Stage, Callable[[bytes], Any]] = {
                Stage.CONVERT: lambda data: self.decode_convert(data),
                Stage.COPY: lambda data: self.decode_copy(data),
                Stage.MAXPOOL: lambda data: self.decode_maxpool(data),
                Stage.MYRIADXHWOP: lambda data: self.decode_hwop(data),
                Stage.PERMUTE: lambda data: self.decode_permute(data),
                Stage.RELU: lambda data: self.decode_relu(data),
                Stage.SCALESHIFT: lambda data: self.decode_scaleshift(data),
                Stage.SOFTMAX: lambda data: self.decode_softmax(data),
                Stage.SUM: lambda data: self.decode_sum(data),
            }
            ########### COPIED FROM stage_decoder.py ##########
            mapper[injected_stage_type](reader.buffer[reader.ptr+8:reader.ptr+injected_stage_len-16])
            # self.decode_copy(reader.buffer[reader.ptr+8:reader.ptr+injected_stage_len-16])
            assert(reader.buffer[reader.ptr+injected_stage_len-16:reader.ptr+injected_stage_len-12] == b'\x13\x00\x00\x00'), "Wrong stage end type?"
            assert(reader.buffer[reader.ptr+injected_stage_len-12:reader.ptr+injected_stage_len-8] == b'\x19\xff\x83\x7f'), "Wrong stage split magic?"
            self.printer.print("Post-read inj-stage", loglevel=logging.ERROR)
            
            reader.skip(injected_stage_len - 8) # TODO: correct? already read 8 bytes of that stage, right?
        self.printer.print(f"   # of buffers: {reader.read_uint()}", loglevel=logging.INFO)
        decoded_dims, decoded_data = self.decode(data[reader.ptr:])
        self.printer.print(
            f"[{stype.name}] Dimensions: {decoded_dims[ChannelIndices.INPUT]} x {decoded_dims[indices[ChannelIndices.COEFFS]]} -> {decoded_dims[indices[ChannelIndices.OUTPUT]]}",
            loglevel=logging.DEBUG)

        kernel = {
            'width': kernel_width,
            'height': kernel_height,
            'stride': kernel_stride
        }
        if stype == OpType.CONV:
            self.handle_conv(kernel, outnum_chans, vecsize, with_relu, decoded_data, indices, decoded_dims, source_x)

        if stype == OpType.POOL and pooltype == PoolType.MAXPOOL:
            self.handle_maxpool(kernel, decoded_dims)

        elif stype == OpType.POOL and pooltype == PoolType.AVGPOOL:
            self.handle_avgpool(kernel, decoded_dims)

        return operations_ret

    def handle_conv(self, kernel: Dict, outnum_chans, vecsize, with_relu, decoded_data, indices, decoded_dims, source_x):
        """
        Handles a hardware-accelerated Convolution operation.
        """
        self.printer.print(
                f"chans: {outnum_chans}, kw: {kernel['width']}, kh: {kernel['height']}, stride: {kernel['stride']}",
                loglevel=logging.DEBUG)

        maybe_dense = False
        if kernel['stride'] == 1:
            self.maybe_split = False
            self.split_shapes = []
        if kernel['stride'] == 2:
            self.printer.print('Convolution with stride=2 detected!', loglevel=logging.WARN)
            self.printer.print('It might be that the following convolutions actually belong together!', loglevel=logging.WARN)
            if self.maybe_split:
                self.printer.print('Last layer was also using stride=2, skipping this layer?', loglevel=logging.WARN)
                self.layer_count['convolution'] += 1
                return

            self.maybe_split = True
            self.split_shapes.append(decoded_dims[0])
            self.printer.print('Add the following shapes together: ', loglevel=logging.ERROR)
            self.printer.print(f'{self.split_shapes}', loglevel=logging.ERROR)
        if kernel['stride'] > 2:
            self.printer.print(f"Convolution with kernel_stride>2: {kernel['stride']}", loglevel=logging.WARN)
            self.printer.print("This might be a Dense layer instead of Conv?!", loglevel=logging.WARN)
            maybe_dense = True

        last_layer_id = self.xml_exporter.layer_id - 1
        last_layer_ports = self.xml_exporter.layer_ports - 1

        batch_size = source_x[0]
        current_chans = source_x[1]
        new_chans = outnum_chans * vecsize
        if not self.is_flattened:
            new_width = source_x[2] // kernel['stride']
            new_height = source_x[3] // kernel['stride']
        else:
            new_width = 1
            new_height = 1

        new_width = max(new_width, 1)
        new_height = max(new_height, 1)

        if self.was_permuted:
            self.printer.print("permute: was permuted = true", loglevel=logging.DEBUG)
            self.printer.print(f"permute: current shape: {self.current_shape}", loglevel=logging.DEBUG)
            self.printer.print(f"permute: kw {kernel['width']} kh {kernel['height']} ks {kernel['stride']}", loglevel=logging.DEBUG)

            # correctly permute
            cs = self.current_shape[::-1]
            pi = self.last_permutation_index
            self.current_shape = (cs[pi[3]], cs[pi[2]], cs[pi[1]], cs[pi[0]])
            self.printer.print(f"permindex ({pi}), reshaped current shape after to {self.current_shape}", loglevel=logging.DEBUG)

            # we want: (1024, 7, 7, 64) for const...
            const_shape = (new_chans, cs[pi[2]], cs[pi[1]], cs[pi[0]])
            self.printer.print(f"DEBUGPERM: {const_shape}", loglevel=logging.DEBUG)
            new_shape = (batch_size, new_chans, new_width, new_height)
            if self.is_flattened:
                new_shape = (new_shape[0], reduce((lambda x, y: x*y), new_shape[1:]))

            # reset last permutation
            self.was_permuted = False
            self.last_permutation = [0,1,2,3]
            kernel_stride_x = self.current_shape[2]//new_width
            kernel_stride_y = self.current_shape[3]//new_height
        if len(self.current_shape) > 2:
            if kernel['width'] > self.current_shape[2]:
                self.printer.print(f"kernelWidth({kernel['width']}) > currentShape({self.current_shape[2]})!", loglevel=logging.WARN)
                kernel['width'] = self.current_shape[2]
            if kernel['height'] > self.current_shape[3]:
                self.printer.print(f"kernelHeight({kernel['height']}) > currentShape({self.current_shape[3]})!", loglevel=logging.WARN)
                kernel['height'] = self.current_shape[3]
            const_shape = (new_chans, current_chans, kernel['width'], kernel['height'])
            new_shape = (batch_size, new_chans, new_width, new_height)
            if self.is_flattened:
                new_shape = (new_shape[0], reduce((lambda x, y: x*y), new_shape[1:]))

            kernel_stride_x = source_x[2]//new_width #=kernelStride?
            kernel_stride_y = source_x[3]//new_height#=kernelStride?
        else:
            const_shape = (new_chans, current_chans)
            new_shape = (batch_size, new_chans)

        self.printer.print(f"current shape {self.current_shape}, so const shape should be {const_shape}", loglevel=logging.INFO)
        self.printer.print(f"new shape is: {new_shape}", loglevel=logging.INFO)

        weights_size = reduce((lambda x, y: x*y), const_shape)
        layer_prefix = f"matmul_{self.layer_count['matmul']}" if maybe_dense and self.experimental_dense_detection else f"convolution_{self.layer_count['convolution']}"
        const_layer = XML.add_const(
            name=f"{layer_prefix}/Weights",
            data={
                'element_type': 'f16',
                'shape': ', '.join(map(str, const_shape)),
                'offset': str(self.bin_exporter.offset),
                'size': str(2*weights_size)
            },
            shape=[const_shape],
            prec="FP16")
        if not maybe_dense:
            # This seems to be problematic!
            self.printer.print(f"input ind: {decoded_dims[ChannelIndices.INPUT]}", loglevel=logging.DEBUG)
            conv_layer = XML.add_conv(
                name=f"{layer_prefix}/Conv2D",
                inshape=self.current_shape,
                weightshape=const_shape,
                outshape=new_shape,
                strides=f'{kernel_stride_x}, {kernel_stride_y}',
                mnist_special_treatment=self.mnist_special_treatment)

        if self.experimental_dense_detection and maybe_dense:
            input_new_n = self.current_shape[0]
            input_new_c = reduce(lambda x, y: x*y, self.current_shape[1:])
            const_n = const_shape[0]
            const_c = reduce(lambda x,y: x*y, const_shape[1:])

            dense_input = (input_new_n, input_new_c)
            const_output = (const_n, const_c)
            dense_output = (input_new_n, const_n)

            self._match_dimensions(dense_input)
            self.is_flattened = True # TODO any case when this is being set to False again?! is there some kind of expand?
            # fix new_shape now that we work with a flattened variant of it
            if len(new_shape) > 2:
                new_shape = (new_shape[0], reduce((lambda x, y: x*y), new_shape[1:]))
            last_layer_id = self.xml_exporter.layer_id - 1
            last_layer_ports = self.xml_exporter.layer_ports - 1
            const_layer['data']['shape'] = ', '.join(map(str, const_output))
            const_layer['data']['offset'] = str(self.bin_exporter.offset)
            const_layer['outputs'][0] = const_output
            
            conv_layer = XML.add_dense(
                name = f"{layer_prefix}/MatMul",
                inshape=dense_input,
                weightshape=const_output,
                outshape=dense_output
            )

        out_shape = tuple(decoded_dims[indices[ChannelIndices.COEFFS]][::-1])
        if not maybe_dense:
            self.bin_exporter.write_bytes(decoded_data[indices[ChannelIndices.COEFFS]], shape = out_shape, scale = self.scale, limit=weights_size, stride = kernel['stride'])
        else:
            self.bin_exporter.write_bytes(decoded_data[indices[ChannelIndices.COEFFS]], shape = out_shape, scale = self.scale, limit=weights_size)
        
            
        self.current_shape = new_shape
        self.printer.print(f"changed current shape to {self.current_shape}", loglevel=logging.INFO)

        const_layer_id, const_layer_ports = self.xml_exporter.add_layer(const_layer)
        conv_layer_id, conv_layer_ports = self.xml_exporter.add_layer(conv_layer)

        prev_to_conv_edge = XEdge(
            XLayer(layer_id = last_layer_id, port_id = last_layer_ports),
            XLayer(layer_id = conv_layer_id, port_id = conv_layer_ports['input'][0])
        )
        self.xml_exporter.add_edge(prev_to_conv_edge)

        const_to_conv_edge = XEdge(
            XLayer(layer_id = const_layer_id, port_id = const_layer_ports['output'][0]),
            XLayer(layer_id = conv_layer_id, port_id = conv_layer_ports['input'][1])
        )
        self.xml_exporter.add_edge(const_to_conv_edge)

        if indices[ChannelIndices.BIASES] > 0:
            bias_size = self.current_shape[1]
            bias_shape = (1, bias_size, 1, 1) if not self.is_flattened else (1, bias_size)
            bias_values_layer = XML.add_const(
                name=f"{layer_prefix}/Biases",
                data={
                    'element_type': 'f16',
                    'shape': ', '.join(map(str, bias_shape)),
                    'offset': str(self.bin_exporter.offset),
                    'size': str(bias_size*2)
                },
                shape=[bias_shape],
                prec="FP16")
            const_layer_id, const_layer_ports = self.xml_exporter.add_layer(bias_values_layer)
            self.bin_exporter.write_bytes(decoded_data[indices[ChannelIndices.BIASES]][:bias_size], scale = self.scale, shape=(1, bias_size, 1, 1))
            biases_layer = XML.add_addition(
                name=f"{layer_prefix}/AddBiases",
                shape=new_shape,
                add_shape=bias_shape)
            bias_layer_id, bias_layer_ports = self.xml_exporter.add_layer(biases_layer)
            bias_edge0 = XEdge( # Conv result is first port, biases second port
                XLayer(layer_id = conv_layer_id, port_id = conv_layer_ports['output'][0]),
                XLayer(layer_id = bias_layer_id, port_id = bias_layer_ports['input'][0])
            )
            self.xml_exporter.add_edge(bias_edge0)
            bias_edge1 = XEdge( # Conv result is first port, biases second port
                XLayer(layer_id = const_layer_id, port_id = const_layer_ports['output'][0]),
                XLayer(layer_id = bias_layer_id, port_id = bias_layer_ports['input'][1])
            )
            self.xml_exporter.add_edge(bias_edge1)
        else: # re-wire layer id and port in case no bias is being added
            bias_layer_id = conv_layer_id
            bias_layer_ports = conv_layer_ports
        if with_relu:
            self.printer.print(decoded_dims[indices[ChannelIndices.OUTPUT]])
            if self.is_flattened and len(new_shape) > 2:
                new_shape = (new_shape[0], reduce((lambda x, y: x*y), new_shape[1:]))
            relu_layer = XML.add_relu(f"{layer_prefix}/ReLU", new_shape)
            relu_layer_id, relu_layer_ports = self.xml_exporter.add_layer(relu_layer)
            # assume that input to ReLU is just the previously added bias addition layer
            relu_edge = XEdge(
                XLayer(layer_id = bias_layer_id, port_id = bias_layer_ports['output'][0]),
                XLayer(layer_id = relu_layer_id, port_id = relu_layer_ports['input'][0])
            )
            self.xml_exporter.add_edge(relu_edge)
        if self.experimental_dense_detection and maybe_dense:
            self.layer_count['matmul'] += 1
        else:
            self.layer_count['convolution'] += 1

    def handle_maxpool(self, kernel: Dict, decoded_dims):
        """
        Handles a hardware-accelerated MaxPool operation.
        """
        # TODO check if this is always constant!
        pool_params = {
            'strides': f"{kernel['stride']}, {kernel['stride']}",
            'pads_begin': '0, 0',
            'pads_end': '0, 0',
            'kernel': f"{kernel['width']}, {kernel['height']}",
            'rounding_type': 'ceil', # remove this for MNIST
            'auto_pad': 'explicit' # seems this is only 'same_upper' for the MNIST model. other models use 'explicit'
            }
        pool_layer = XML.add_pooling(
            f"pooling_{self.layer_count['pooling']}/MaxPool",
            pool_params,
            decoded_dims[0],
            decoded_dims[1])
        layer_id, _ = self.xml_exporter.add_layer(pool_layer) # layer ports unused here?
        pooling_edge =  XEdge(
            XLayer(layer_id = layer_id - 1, port_id = 1),
            XLayer(layer_id = layer_id, port_id = 0)
        )
        # we just assume that pooling layer has the last added layer:port1 as input (most likely ReLU)
        self.xml_exporter.add_edge(pooling_edge)
        self.current_shape = decoded_dims[1]
        self.printer.print(f"reshaped current shape after MaxPooling to: {self.current_shape}", loglevel=logging.INFO)
        self.layer_count['pooling'] += 1

    def handle_avgpool(self, kernel: Dict, decoded_dims):
        """
        Handles a hardware-accelerated AvgPool operation.
        """
        pool_params = {
            'auto_pad': 'same_upper',
            'kernel': f"{kernel['width']},{kernel['height']}",
            'pads_begin': '0,0',
            'pads_end': '0,0',
            'strides': f"{kernel['stride']},{kernel['stride']}"
        }
        pool_layer = XML.add_avg_pooling(
            f"pooling_{self.layer_count['pooling']}/AvgPool",
            pool_params,
            decoded_dims[0],
            decoded_dims[1])
        layer_id, _ = self.xml_exporter.add_layer(pool_layer)
        pooling_edge = XEdge(
            XLayer(layer_id = layer_id - 1, port_id = 1),
            XLayer(layer_id = layer_id, port_id = 0)
        )
        self.xml_exporter.add_edge(pooling_edge)
        self.current_shape = decoded_dims[1]
        self.printer.print(f"Reshaped current shape after AvgPooling to: {self.current_shape}", loglevel=logging.INFO)
        self.layer_count['pooling'] += 1

    def decode_permute(self, data):
        """
        Decodes Permute stage.
        """
        reader = DataReader(data)
        perm_index = reader.read_int(count = 8)
        try:
            n_permutations = perm_index.index(-1)
        except ValueError:
            n_permutations = -1
            pass
        self.printer.print(f"Identified Permute stage (indices: {perm_index})", loglevel=logging.WARN)
        self.printer.print(f"We only care about the first {n_permutations} indices since all other are -1.", loglevel=logging.WARN)
        dims, _ = self.decode(data[reader.ptr:])
        self.printer.print(f"Permute: {dims[0]} -> {dims[1]}", loglevel=logging.WARN)
        last_layer_id = self.xml_exporter.layer_id - 1
        last_layer_port_count = self.xml_exporter.layer_ports - 1
        # Step 1: create Const stage
        constdata = {
            'element_type': 'i64',
            'shape': '4',
            'offset': str(self.bin_exporter.offset),
            'size': str(len(dims[0])*8)
        }
        permutation = [int(x).to_bytes(8, byteorder="little") for x in perm_index[:n_permutations]]
        materialized_permutation = permutation[1] + permutation[0] + permutation[3] + permutation[2]
        # TODO understand why the permutation itself needs to be permuted?!
        self.bin_exporter.write_bytes(materialized_permutation)

        const_layer = XML.add_const(f"transpose_{self.layer_count['transpose']}/Indices", constdata, [(4,)], 'I64')
        const_layer_id, const_layer_ports = self.xml_exporter.add_layer(const_layer)

        # Step 2: create Transpose stage
        transpose_layer = XML.add_transpose(f"transpose_{self.layer_count['transpose']}", [dims[0], (4,)], [dims[1]])
        transpose_layer_id, transpose_layer_ports = self.xml_exporter.add_layer(transpose_layer)

        # Step 3: add edges
        transpose_edge0 = XEdge(
            XLayer(layer_id = last_layer_id, port_id = last_layer_port_count),
            XLayer(layer_id = transpose_layer_id, port_id = transpose_layer_ports['input'][0])
        )
        self.xml_exporter.add_edge(transpose_edge0)

        transpose_edge1 = XEdge(
            XLayer(layer_id = const_layer_id, port_id = const_layer_ports['output'][0]),
            XLayer(layer_id = transpose_layer_id, port_id = transpose_layer_ports['input'][1])
        )
        self.xml_exporter.add_edge(transpose_edge1)

        # TODO permutation should work like this, but this confuses following Convolution?!
        #print(f"current shape: {self.current_shape}, dims: {dims}, permIndex: {permIndex}")
        cs = self.current_shape[::-1]
        pi = perm_index[::-1]
        #self.current_shape = (cs[pi[0]], cs[pi[1]], cs[pi[2]], cs[pi[3]])
        self.was_permuted = True
        self.last_permutation_index = perm_index

        self.layer_count['transpose'] += 1

    def decode_copy(self, data):
        """
        Decode Copy stage.
        """
        self.printer.print("Identified Copy stage (no parameters).", loglevel=logging.INFO)
        self.decode(data, in_out_desc=('IN', 'OUT'))

    def decode_scaleshift(self, data):
        """
        Decode ScaleShift stage.
        """
        self.printer.print("Identified ScaleShift stage (no parameters).", loglevel=logging.INFO)
        shapes, dat = self.decode(data, in_out_desc=('IN', 'OUT', 'IN (SCALE)', 'IN (SHIFT)'))

        scales = np.array([dat[2][i][0] for i in range(len(dat[2]))], dtype=np.float16)
        shifts = np.array([dat[3][i][0] for i in range(len(dat[3]))], dtype=np.float16)
        self.printer.print(f"Scale: {scales}", loglevel=logging.DEBUG)
        self.printer.print(f"Shift: {shifts}", loglevel=logging.DEBUG)

        if np.array_equal(scales, np.ones(shape=scales.shape)):
            self.printer.print("All scales are 1.0. Setting self.scale to 1.0", loglevel=logging.INFO)
            self.scale = np.float16(1.0)

        scale_shift_const_layer = XML.add_const(
            name=f"add_{self.layer_count['add']}/Constant",
            data={
                'element_type': 'f16',
                'shape': f'1, {len(shifts)}, 1, 1',
                'offset': str(self.bin_exporter.offset),
                'size': str(len(shifts) * 2)
            },
            shape=[(1,len(shifts),1,1)],
            prec='FP16'
        )

        origin_id = self.xml_exporter.layer_id
        origin_ports = self.xml_exporter.layer_ports

        const_layer_id, const_layer_ports = self.xml_exporter.add_layer(scale_shift_const_layer)

        self.bin_exporter.write_bytes(bytes(shifts))
        self.printer.print(f"Wrote shifts to bin file: {shifts}", loglevel=logging.INFO)


        scale_shift_layer = XML.add_addition(
            name=f"add_{self.layer_count['add']}",
            shape=self.current_shape,
            add_shape=[1,len(shifts),1,1]
        )

        add_layer_id, add_layer_ports = self.xml_exporter.add_layer(scale_shift_layer)

        edge_input_add = XEdge(
            l_from=XLayer(
                layer_id=origin_id - 1,
                port_id=origin_ports - 1
            ),
            l_to=XLayer(
                layer_id=add_layer_id,
                port_id=add_layer_ports['input'][0]
            )
        )
        self.xml_exporter.add_edge(edge_input_add)

        edge_const_add = XEdge(
            l_from=XLayer(
                layer_id=const_layer_id,
                port_id=const_layer_ports['output'][0]),
            l_to=XLayer(
                layer_id=add_layer_id,
                port_id=add_layer_ports['input'][1])
        )
        self.xml_exporter.add_edge(edge_const_add)
        self.layer_count['add'] += 1

    def decode_relu(self, data):
        """
        Decodes Relu stage.
        """
        self.printer.print("Identified Relu stage (not implemented).", loglevel=logging.ERROR)
        shapes, dat = self.decode(data, in_out_desc=('IN', 'OUT'))

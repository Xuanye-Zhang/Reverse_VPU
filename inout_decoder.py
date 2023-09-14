"""
Decodes the input and output section of stages of the USB BLOB.
"""

from functools import reduce
import logging
import struct
from io import BytesIO
from typing import Tuple
import numpy as np

from util import read_uint
from constants import DATA_LOCATION, DATA_TYPE, DimsOrder


class InOutDecoder:
    """
    TODO
    """

    def __init__(self, cbuffer, printer):
        self.buf = cbuffer
        self.counter = 0
        self.printer = printer

    def __repr__(self):
        return 'InOutDecoder'

    def _read_location(self, perm_size):
        """
        Reads location info (either dimension or stride).
        Returns: - data location
                 - data offset
                 - current location
        """

        get_uint = lambda: read_uint(self.data)

        location = DATA_LOCATION[get_uint()]
        offset = int(get_uint())

        shape_product = 1
        if location == 'blob':
            shape = ()
            for i in range(perm_size):
                values = struct.unpack('I', self.buf[offset+i*4:offset+i*4+4])[0]
                shape += (values,)
                shape_product *= values
                if i > 32:
                    break
        
        return location, offset, shape, shape_product

    def decode_one_channel(self):
        """
        Decodes a single channel.
        Returns: - channel shape
                 - data_location
                 - data_offset
                 - raw data (if location is blob)
        """
        get_uint = lambda: read_uint(self.data)

        self.printer.print(f"    Channel #{self.counter}:", loglevel=logging.INFO)
        self.printer.print(f"        storedDesc.type = {DATA_TYPE[get_uint()]}", loglevel=logging.INFO)
        dims_order_code = get_uint()
        dims_order = DimsOrder(dims_order_code)
        self.printer.print(f"        storedDimsOrder.code = {dims_order.name} ({hex(dims_order_code)})", loglevel=logging.INFO)
        perm_size = int(get_uint())
        self.printer.print(f"        storedPerm.size = {hex(perm_size)}", loglevel=logging.INFO)
        
        dim_location, dim_offset, dim_shape, dims_product = self._read_location(perm_size)
        stride_location, stride_offset, stride_shape, _ = self._read_location(perm_size)

        def interpret_shape_by_dimorder_code(dim_shape, perm_size, dims_code):
            assert(len(dim_shape) == perm_size), f"length of shape ({len(dim_shape)}) does not match permutation size ({perm_size})!"
            assert(isinstance(dims_code, DimsOrder)), f"dims_code is not an instance of type DimsOrder but of type {type(dims_code)}!"
            if dims_code == DimsOrder.NCHW: # 0x4321
                return dim_shape[::-1]
            elif dims_code == DimsOrder.C: # 0x3
                return (1, dim_shape[0], 1, 1)
            elif dims_code == DimsOrder.NC: # 0x43
                return dim_shape[::-1]
            elif dims_code == DimsOrder.CHW: # 0x321
                return dim_shape[::-1]
            else:
                input('Other shape. Press enter to continue')
            return dim_shape

        adjusted_dim_shape = interpret_shape_by_dimorder_code(dim_shape, perm_size, dims_order)
        self.printer.print(
            f"        shapeLocation.dims = {dim_location}@{hex(dim_offset)}, shape: {adjusted_dim_shape}",
            loglevel=logging.INFO)        
        self.printer.print(
            f"        shapeLocation.strides = {stride_location}@{hex(stride_offset)}, shape: {stride_shape}",
            loglevel=logging.INFO)
        data_location = DATA_LOCATION[get_uint()]
        offset = 0
        io_idx = None
        parent_byte_size = None
        if data_location in ('input', 'output'):
            io_idx = get_uint()
            parent_byte_size = get_uint()
            offset = 8
        data_offset = int(get_uint())

        info_string = f"(ioIdx: {io_idx}, parentSize: {parent_byte_size})" if offset > 0 else ""

        data_extra = ""
        retdata = None
        if data_location == 'blob':
            data_extra = "("
            retdata = []
            for i in range(dims_product):
                tmp = np.frombuffer(
                    buffer=self.buf[data_offset+i*2:data_offset+i*2+2],
                    dtype=np.float16)
                retdata.append(tmp)
            data_extra += ")"
        self.printer.print(
            f"        dataLocation: {data_location}@{hex(data_offset)} {info_string}",
            loglevel=logging.DEBUG)
        self.counter += 1

        return adjusted_dim_shape, data_location, data_offset, retdata


    def _reset_channel_counter(self):
        """
        Resets channel counter.
        Is being called by `decode_inouts`.
        """
        self.counter = 0

    def _set_data(self, data):
        """
        Sets data array.
        Is being called by `decode_inouts`.
        """
        self.data = data

    def decode_inouts(self, data, in_out_desc: Tuple = None, inshape = None):
        """
        Decodes all channels for a given stage.
        """
        data_len = len(data)
        data = BytesIO(data)
        self.printer.print( "  Data:", loglevel=logging.DEBUG)

        return_dimensions = []
        retval = []

        self._reset_channel_counter()
        self._set_data(data)
        write_candidates = []
        idx = 0
        while data.tell() < data_len:
            dim, dimloc, doff, retv = self.decode_one_channel()
            mode = "UNKNOWN"
            if dimloc in ('input', 'blob', 'output') or self.counter == 1:
                mode = "READ: "
            elif dimloc in ('cmx', 'bss'):
                mode = "WRITE:"
                write_candidates.append(self.counter-1)
            if in_out_desc and len(in_out_desc) > idx:
                mode = in_out_desc[idx]
                idx += 1
            start = doff
            shape = dim
            length = reduce(lambda x, y: x*y, shape) * 2
            self.printer.print(
                f'{mode} {dimloc}@[0x{start:08x}-0x{(start+length):08x}] (shape: {shape}/inshape: {inshape})',
                loglevel=logging.ERROR)
            return_dimensions.append(dim)
            retval.append(retv)
        if len(write_candidates) == 1:
            self.printer.print(f"Channel #{write_candidates[0]} is most probably the write channel!", loglevel=logging.INFO)
        else:
            self.printer.print(f"Not sure about write, could be: {write_candidates}", loglevel=logging.INFO)
        return return_dimensions, retval

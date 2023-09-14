"""
Interprets stage section.
"""

from io import BytesIO
import logging
import traceback
from typing import Any, Callable, Dict
from constants import Stage
from decode_stage import StageDecoder
from export_to_xml import XMLExporter
from print_helper import PrintHelper
from util import read_uint

class StageSectionInterpreter:
    """
    TODO
    """

    def __init__(self, data: BytesIO, cbuf: BytesIO, outfile: str, layer_names: Dict[str, str], printer: PrintHelper, xml_exporter: XMLExporter, mnist: bool):
        self.data = data
        self.printer = printer
        self.cbuf = cbuf
        self.xml_exporter = xml_exporter
        self.decoder = StageDecoder(
            out_bin=outfile+'.bin',
            out_xml=outfile+'.xml',
            printer=printer,
            cbuf=cbuf,
            layer_names=layer_names,
            xml_exporter=self.xml_exporter,
            mnist=mnist)
        self.layer_names = layer_names
        self.stage_list = []
        self.stage_counter = dict((stage_name, 0) for stage_name, _ in Stage.__members__.items())
        self.stage_counter['total'] = 0

        self.mapper: Dict[str, Callable[[bytes], Any]] = {
            Stage.CONVERT: lambda data: self.decoder.decode_convert(data),
            Stage.COPY: lambda data: self.decoder.decode_copy(data),
            Stage.MAXPOOL: lambda data: self.decoder.decode_maxpool(data),
            Stage.MYRIADXHWOP: lambda data: self.decoder.decode_hwop(data),
            Stage.PERMUTE: lambda data: self.decoder.decode_permute(data),
            Stage.RELU: lambda data: self.decoder.decode_relu(data),
            Stage.SCALESHIFT: lambda data: self.decoder.decode_scaleshift(data),
            Stage.SOFTMAX: lambda data: self.decoder.decode_softmax(data),
            Stage.SUM: lambda data: self.decoder.decode_sum(data),
        }

    def decode_stage(self, stage_type: int, stage_data: bytes):
        """
        Decodes stage and performs logging tasks.
        """
        try:
            stage = Stage(stage_type)

            self.stage_list.append(stage.name)
            self.stage_counter[stage.name] += 1
        except BaseException as err:
            self.printer.print(f'Unknown stage type: {hex(stage_type)}', loglevel=logging.WARN)
            self.printer.print(f'Stage params + data {stage_data}', loglevel=logging.WARN)
            self.stage_list.append('UNKNOWN')
            pass

        try:
            sub_ops = self.mapper[stage](stage_data)
            if sub_ops:
                self.printer.print(f'Adding sub operation {sub_ops}', loglevel=logging.INFO)
                self.stage_list.append(sub_ops)
        except BaseException as err:
            self.printer.print(f"Unexpected {err=}, {type(err)=}", loglevel=logging.ERROR)
            print(traceback.format_exc())
            if isinstance(err, SystemExit):
                raise
        self.stage_counter['total'] += 1

    def finalize(self):
        """
        Cleans up.
        """
        self.printer.print(self.stage_counter, loglevel=logging.INFO)
        self.printer.print(self.stage_list, loglevel=logging.INFO)
        self.decoder.finalize()



def interpret_stage_section(
    data: BytesIO,
    num_stages: int,
    printer: PrintHelper,
    outfile: str,
    cbuf: BytesIO,
    net_name: str,
    layer_names: Dict[str, str],
    interactive: bool,
    mnist: bool) -> int:
    """
    Interprets stage section.
    """
    get_uint: Callable[[None], int] = lambda: read_uint(data)
    printer.print("Stage section start.", loglevel=logging.INFO)
    xml_exporter = XMLExporter(net_name=net_name, layer_names=layer_names)
    interpreter = StageSectionInterpreter(
        data=data,
        cbuf=cbuf,
        outfile=outfile,
        layer_names=layer_names,
        printer=printer,
        xml_exporter=xml_exporter,
        mnist=mnist)

    for i in range(num_stages):
        if interactive:
            input('Press enter to continue')
        stage_length = get_uint()
        printer.print(f"Processing stage {i+1}/{num_stages}.", loglevel=logging.INFO)
        printer.print(f"Stage length {stage_length}", inc=4, loglevel=logging.DEBUG)
        stage_type = get_uint()
        try:
            stage_type = Stage(stage_type)
            printer.print(f"Stage Type {stage_type.name} ({hex(stage_type.value)})", inc=4, loglevel=logging.INFO)
        except BaseException as err:
            printer.print(f"Unexpected err={err}, type(err)={type(err)}", loglevel=logging.WARN)
            printer.print(f"Stage Type UNKNOWN ({hex(stage_type)})", inc=4, loglevel=logging.WARN)

        printer.print(f"# SHAVEs {get_uint()}", inc=4, loglevel=logging.DEBUG)
        printer.print(f"Params pos {get_uint()}", inc=4, loglevel=logging.DEBUG)
        stage_data = data.read(stage_length - 24)
        printer.set_pos(printer.get_pos() + (stage_length - 24))

        interpreter.decode_stage(stage_type, stage_data)

        stage_type_end = get_uint()
        printer.print(f"Stage end type {hex(stage_type_end)}", inc=4, loglevel=logging.DEBUG)
        stage_split_magic = get_uint()
        printer.print(f"Stage split magic {hex(stage_split_magic)}", inc=4, loglevel=logging.DEBUG)

    printer.print("Stage section end.", loglevel=logging.INFO)
    interpreter.finalize()
    return printer.get_pos()

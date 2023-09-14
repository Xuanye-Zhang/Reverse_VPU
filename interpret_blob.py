"""
Interpret entire USB BLOB.
"""

from io import BytesIO
import logging
import sys

from constants import ELF_HEADER_START, ELF_HEADER_END, MV_HEADER_START, MV_HEADER_END
from in_section import interpret_input_section
from const_data_section import interpret_constdata_section
from out_section import interpret_output_section
from stage_section import interpret_stage_section
from headers.interpret_headers import read_elf, read_mv
from print_helper import PrintHelper

def interpret_blob(data: bytes, verbose: bool, outfile: str, net_name: str, interactive: bool, mnist: bool):
    """
    Interpret entire USB BLOB.
    It starts with ELF and MV header.
    Then, the different sections are being interpreted.
    """
    error = None
    loglevel = logging.DEBUG if verbose else logging.INFO
    printer = PrintHelper(loglevel=loglevel)
    printer.print("==========")
    read_elf(BytesIO(data[ELF_HEADER_START:ELF_HEADER_END]), printer)
    if interactive:
        input('Press enter to continue')

    printer.print("==========")
    sec, inputs, outputs, stages, filesize = read_mv(BytesIO(data[MV_HEADER_START:MV_HEADER_END]), printer)

    if filesize > len(data):
        printer.print(f"Header filesize ({filesize}) is larger than actual file size ({len(data)})!", loglevel=logging.ERROR)
        error = True
    if error:
        printer.print("One or more error(s) occured. Quitting.", loglevel=logging.ERROR)
        sys.exit(0)
    
    input_buffer = BytesIO(data[sec['in']:sec['out']])
    output_buffer = BytesIO(data[sec['out']:sec['stages']])
    stage_buffer = BytesIO(data[sec['stages']:sec['const_data']])
    const_buffer = BytesIO(data[sec['const_data']:])
    if interactive:
        input('Press enter to continue')

    printer.print("==========")
    printer.print(f"Skipping padding from {printer.get_pos()} to {sec['in']}")
    printer.set_pos(sec['in'])
    offset_shapes_in, offset_strides_in, input_name = interpret_input_section(
        data=input_buffer,
        input_section=sec['in'],
        num_inputs=inputs,
        printer=printer)
    if offset_shapes_in > len(data):
        printer.print(f"Input shape offset ({offset_shapes_in}) is larger than file length ({len(data)})!", loglevel=logging.ERROR)
        error = True
    if offset_strides_in > len(data):
        printer.print(f"Input stride offset ({offset_strides_in}) is larger than file length ({len(data)})!", loglevel=logging.ERROR)
        error = True
    if error:
        printer.print("One or more error(s) occured. Quitting.", loglevel=logging.ERROR)
        sys.exit(1)
    if interactive:
        input('Press enter to continue')

    printer.print("==========")
    printer.print(f"Skipping padding from {printer.get_pos()} to {sec['out']}")
    printer.set_pos(sec['out'])
    offset_shapes_out, offset_strides_out, output_name = interpret_output_section(
        data=output_buffer,
        output_section=sec['out'],
        num_outputs=outputs,
        printer=printer)
    if offset_shapes_out > len(data):
        printer.print(f"Output shape offset ({offset_shapes_out}) is larger than file length ({len(data)})!", loglevel=logging.ERROR)
        error = True
    if offset_shapes_out > len(data):
        printer.print(f"Output stride offset ({offset_strides_out}) is larger than file length ({len(data)})!", loglevel=logging.ERROR)
        error = True
    if error:
        printer.print("One or more error(s) occured. Quitting.", loglevel=logging.ERROR)
        sys.exit(1)
    if interactive:
        input('Press enter to continue')

    printer.print("==========")
    printer.print(f"Skipping padding from {printer.get_pos()} to {sec['stages']}")
    printer.set_pos(sec['stages'])
    interpret_stage_section(
        data=stage_buffer,
        num_stages=stages,
        printer=printer,
        outfile=outfile,
        cbuf=const_buffer,
        net_name=net_name,
        layer_names={
            'input': input_name,
            'output': output_name
        },
        interactive=interactive,
        mnist=mnist)
    if interactive:
        input('Press enter to continue')

    printer.print("==========")
    printer.print(f"Skipping padding from {printer.get_pos()} to {sec['const_data']}")
    printer.set_pos(sec['const_data'])
    interpret_constdata_section(const_buffer, printer)

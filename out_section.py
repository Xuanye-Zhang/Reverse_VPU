"""
Decodes output section.
"""

from io import BytesIO
import logging
from typing import Tuple
from constants import DATA_LOCATION, DATA_TYPE, DimsOrder
from print_helper import PrintHelper
from util import read_uint


def interpret_output_section(
    data: BytesIO,
    output_section: int,
    num_outputs: int,
    printer: PrintHelper) -> Tuple[int, int, str]:
    """
    Decodes output section.
    Returns: offsets for shape dimension and stride and output name.
    """
    get_uint = lambda: read_uint(data)
    printer.set_pos(output_section)
    printer.print("Output section start.", loglevel=logging.INFO)
    for _ in range(num_outputs):
        printer.print(f"Output index {get_uint()}", inc=4, loglevel=logging.INFO)
        printer.print(f"Output offset {get_uint()}", inc=4, loglevel=logging.INFO)
        name_len = get_uint()
        printer.print(f"Name len {name_len}", inc=4, loglevel=logging.DEBUG)
        name = data.read(name_len)
        printer.print(f"Name '{name.decode()}' [raw bytes: {name}]", inc=name_len, loglevel=logging.INFO)
        printer.print(f"Shape Type {DATA_TYPE[get_uint()]}", inc=4, loglevel=logging.DEBUG)
        shape = get_uint()
        printer.print(f"Shape Code {DimsOrder(shape).name} ({hex(shape)})", inc=4, loglevel=logging.INFO)
        printer.print(f"Shape Size {get_uint()}", inc=4, loglevel=logging.DEBUG)
        printer.print(f"Shape Dims Location {DATA_LOCATION[get_uint()]}", inc=4, loglevel=logging.DEBUG)
        shape_dims_offset = get_uint()
        printer.print(f"Shape Dims Offset {hex(shape_dims_offset)}", inc=4, loglevel=logging.DEBUG)
        printer.print(f"Shape Strides Location {DATA_LOCATION[get_uint()]}", inc=4, loglevel=logging.DEBUG)
        shape_strides_offset = get_uint()
        printer.print(f"Shape Strides Offset {hex(shape_strides_offset)}", inc=4, loglevel=logging.DEBUG)
    printer.print("Output section end.", loglevel=logging.INFO)
    return shape_dims_offset, shape_strides_offset, name.decode().rstrip('\x00')

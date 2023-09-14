"""
Decodes input section.
"""

from io import BytesIO
import logging
from typing import Tuple
from constants import DATA_LOCATION, DATA_TYPE, DimsOrder
from print_helper import PrintHelper
from util import read_uint


def interpret_input_section(
    data: BytesIO,
    input_section: int,
    num_inputs: int,
    printer: PrintHelper) -> Tuple[int, int, str]:
    """
    Decodes input section.
    Returns: offsets for shape dimension and stride and input name.
    """
    get_uint = lambda: read_uint(data)
    printer.set_pos(input_section)
    printer.print("Input section start.", loglevel=logging.INFO)
    for _ in range(num_inputs):
        printer.print(f"Input index {get_uint()}", inc=4, loglevel=logging.INFO)
        printer.print(f"Input offset {get_uint()}", inc=4, loglevel=logging.INFO)
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
        printer.print(f"Shape Strides Offset ({hex(shape_strides_offset)})", inc=4, loglevel=logging.DEBUG)
    printer.print("Input section end.", loglevel=logging.DEBUG)
    return shape_dims_offset, shape_strides_offset, name.decode().rstrip('\x00')

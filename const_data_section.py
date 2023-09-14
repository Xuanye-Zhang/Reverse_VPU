"""
Decodes constant data section.
"""

from io import BytesIO
import logging
import struct
from print_helper import PrintHelper


def interpret_constdata_section(data: BytesIO, printer: PrintHelper):
    """
    Interprets the constant data section.
    Outputs raw byte values interpreted as integers, floats and their hex value.
    Only for debugging purposes.
    """
    constdata_counter = 0
    printer.print("ConstData section start.", loglevel=logging.INFO)
    for dat in iter(lambda: data.read(4), b''):
        dat_int = struct.unpack('I', dat)[0]
        dat_float = struct.unpack('f', dat)[0]
        printer.print(f"{dat_int} (float: {dat_float} , hex: {dat.hex()})", loglevel=logging.DEBUG, inc=4, callerlog=False)
        constdata_counter += 1
    printer.print("ConstData section end.", loglevel=logging.INFO)
    printer.print(f"Number of const data entries: {constdata_counter}", loglevel=logging.INFO)
    return printer.get_pos()

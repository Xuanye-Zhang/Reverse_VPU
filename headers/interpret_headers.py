"""
Interpret ELF and MV header from USB BLOB file.
"""

from io import BytesIO
import logging
from typing import Dict, Tuple
from print_helper import PrintHelper
from util import read_short, read_uint


def read_elf(data: BytesIO, printer: PrintHelper):
    """
    Reads ELF header from USB BLOB and prints information.
    The ELF header is 52 bytes long, so provide a buffer with at least 52 bytes content.
    """
    get_uint = lambda: read_uint(data)
    get_short = lambda: read_short(data)
    printer.print("ELF Header Start", loglevel=logging.INFO)
    printer.print(f"Signature: {data.read(16)}", inc=16, loglevel=logging.DEBUG)
    printer.print(f"Type: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print(f"Machine: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print(f"Version: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Entry: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"PH Offset: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"SH Offset: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Flags: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Header Size: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print(f"Program Header Entry Size: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print(f"Program Header Entry Num: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print(f"Section Header Entry Size: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print(f"Section Header Entry Num: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print(f"Section Header Index: {get_short()}", inc=2, loglevel=logging.DEBUG)
    printer.print("ELF Header End", loglevel=logging.INFO)


def read_mv(data: BytesIO, printer: PrintHelper) -> Tuple[Dict[str, int], int, int, int, int]:
    """
    Reads MV header from USB BLOB and prints information.
    The ELF header is 80 bytes long, so provide a buffer with at least 80 bytes content.
    Returns: - position of sections
             - number of inputs/outputs/stages
             - file size according to header
    """
    get_uint = lambda: read_uint(data)
    printer.print("MV Header Start", loglevel=logging.INFO)
    printer.print(f"BLOB Magic Number: {hex(get_uint())}", inc=4, loglevel=logging.DEBUG)
    filesize = get_uint()
    printer.print(f"File Size: {filesize}", inc=4, loglevel=logging.INFO)
    printer.print(f"BLOB Version Major: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"BLOB Version Minor: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    num_in, num_out, num_stages = get_uint(), get_uint(), get_uint()
    printer.print(f"Number of inputs: {num_in}", inc=4, loglevel=logging.INFO)
    printer.print(f"Number of outputs: {num_out}", inc=4, loglevel=logging.INFO)
    printer.print(f"Number of stages: {num_stages}", inc=4, loglevel=logging.INFO)
    printer.print(f"Input size: {get_uint()}", inc=4, loglevel=logging.INFO)
    printer.print(f"Output size: {get_uint()}", inc=4, loglevel=logging.INFO)
    printer.print(f"Batch size: {get_uint()}", inc=4, loglevel=logging.INFO)
    printer.print(f"BSS Mem Size: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"CMX Slices: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"SHAVEs: {get_uint()}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Has HW Stage: {bool(get_uint())}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Has SHAVE Stage: {bool(get_uint())}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Has DMA Stage: {bool(get_uint())}", inc=4, loglevel=logging.DEBUG)

    in_sec, out_sec, stage_sec, const_sec = get_uint(), get_uint(), get_uint(), get_uint()
    sections = {
        'in': in_sec,
        'out': out_sec,
        'stages': stage_sec,
        'const_data': const_sec
    }
    printer.print(f"Input Section offset: {in_sec}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Output Section offset: {out_sec}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Stage Section offset: {stage_sec}", inc=4, loglevel=logging.DEBUG)
    printer.print(f"Const Data Section offset: {const_sec}", inc=4, loglevel=logging.DEBUG)
    printer.print("MV Header End", loglevel=logging.INFO)
    return sections, num_in, num_out, num_stages, filesize

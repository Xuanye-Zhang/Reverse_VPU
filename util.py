"""
Utility functions.
"""

import struct


def read_short(data):
    """
    Read one short (16 bit) from `data`.
    """
    return struct.unpack('H', data.read(2))[0]

def read_uint(data):
    """
    Read one unsigned int (32 bit) from `data`.
    """
    return struct.unpack('I', data.read(4))[0]

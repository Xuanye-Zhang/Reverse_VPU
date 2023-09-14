"""
Global constants used in the tool.
"""

from enum import Flag, IntEnum


ELF_HEADER_START = 0
ELF_HEADER_END = 52
MV_HEADER_START = ELF_HEADER_END
MV_HEADER_END = MV_HEADER_START + 80


DATA_TYPE = { 0: 'fp16', 1: 'u8', 2: 's32', 3: 'fp32', 4: 'i8' }
DATA_LOCATION = { 0: 'none', 1: 'input', 2: 'output', 3: 'blob', 4: 'bss', 5: 'cmx' }


class Stage(IntEnum):
    EMPTY = -0x01
    MAXPOOL = 0x01
    SOFTMAX = 0x03
    RELU = 0x06
    SUM = 0x0c
    COPY = 0x13
    PERMUTE = 0x22
    MYRIADXHWOP = 0x26
    SCALESHIFT = 0x2f
    CONVERT = 0x6f

class ChannelIndices(IntEnum):
    """
    Channel indices.
    """
    INPUT = 0
    OUTPUT = 1
    COEFFS = 2
    BIASES = 3
    SCALES = 4

class DimsOrder(IntEnum):
    """
    Dimension Orders
    (taken from inference-engine/src/vpu/graph_transformer/src/model/data_desc.cpp).
    """
    Empty = 0x00 # is this actually used?
    C = 0x03    # most commonly used?
    NC = 0x43
    CHW = 0x321
    HWC = 0x213
    HCW = 0x231
    NCHW = 0x4321 # most commonly used?
    NHWC = 0x4213
    NHCW = 0x4231
    NCDHW = 0x43521
    NDHWC = 0x45213

# TODO: more padmodes are possible.
class PadModes(Flag):
    """
    Possible padmodes.
    """
    PAD_ZEROS = 0
    PAD_REPEAT_RIGHT_EDGE = 1
    PAD_REPEAT_BOTTOM_EDGE = 2
    PAD_REPEAT_TOP_EDGE = 4
    PAD_REPEAT_LEFT_EDGE = 8
    PAD_REPEAT_RIGHT_AND_BOTTOM_EDGE = PAD_REPEAT_RIGHT_EDGE | PAD_REPEAT_BOTTOM_EDGE

class OpModes(IntEnum):
    """
    Possible opmodes.
    """
    MODE_1_256 = 0
    MODE_2_128 = 1
    MODE_4_64  = 2
    MODE_8_32  = 3
    MODE_16_16 = 4

class PoolType(IntEnum):
    """
    Possible pooltypes.
    """
    MAXPOOL = 0
    AVGPOOL = 1

class OpType(IntEnum):
    """
    Possible optypes.
    """
    CONV = 0
    CONVPOOL = 1
    FC = 2
    POOL = 4

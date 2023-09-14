"""
Export model weights to binary file.
"""

import logging
import numpy as np

from print_helper import PrintHelper


class BinExporter:
    """
    Exports weights to binary file.
    """

    def __init__(self, filename, printer: PrintHelper):
        self.bin_file = open(filename, 'wb')
        self.calls = 0
        self.offset = 0
        self.printer = printer

    def __repr__(self):
        return 'BinExporter'

    # shape is NCHW
    # NumBatches
    # Channels
    # Height
    # Width

    def write_bytes(self, data, shape = None, scale = None, limit = None, stride=1):
        """
        Writes bytes from given array `data` to weights file
        in (hopefully) correct order.
        """
        self.calls += 1

        if isinstance(data, bytes):
            pre_pos = self.bin_file.tell()
            self.bin_file.write(data)
            post_pos = self.bin_file.tell()
            diff = post_pos - pre_pos
            self.printer.print(
                f"Wrote {diff} bytes ({len(data)} values) to file.",
                loglevel=logging.INFO)
            self.offset = self.bin_file.tell()
            self.bin_file.flush()
            return

        data = np.float16(data)
        if scale:
            data *= scale
        if shape:
            newdata = np.reshape(data, shape[::-1])

            write_buffer = []
            shape = shape[::-1]
            channels = shape[1] if stride == 1 else shape[1] - 1
            for n in range(shape[0]):
                for w in range(shape[3]):
                    for c in range(channels):
                        for h in range(shape[2]):
                            write_buffer.append(newdata[n][c][h][w])
            if isinstance(limit, int):
                write_buffer = write_buffer[:limit]
            arr = np.array(write_buffer, dtype=np.float16)
            self.bin_file.write(arr)
            self.offset = self.bin_file.tell()
            self.printer.print(f'Wrote {len(write_buffer)} float16 values ({len(write_buffer)*2} bytes) with shape={shape} and stride={stride}.', loglevel=logging.INFO)
        else:
            self.bin_file.write(data)
            self.offset = self.bin_file.tell()
            self.printer.print(f'Wrote {len(data)} float16 values ({len(data)*2} bytes) without given shape.', loglevel=logging.WARN)
        self.bin_file.flush()

    def finalize(self):
        """
        Flush buffer and close file.
        """
        num_bytes = self.bin_file.tell()
        self.bin_file.close()
        self.printer.print(f"Wrote {num_bytes} to {self.bin_file.name} in total.", loglevel=logging.INFO)

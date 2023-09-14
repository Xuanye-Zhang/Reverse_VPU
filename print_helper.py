"""
PrintHelper class to easily enable/disable output and keep track of current offset.
"""

import logging
from colorlog import ColoredFormatter
import inspect
from typing import Optional

class PrintHelper:
    """
    Helper class to quickly disable output.
    Uncomment to enable verbose output.
    """
    def __init__(self, loglevel: int):
        self.offset = 0
        self.loglevel = loglevel

        formatter = ColoredFormatter(
            "%(log_color)s%(levelname)-8s%(reset)s | %(log_color)s%(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                'DEBUG': 'white',
                'INFO': 'cyan',
                'WARNING': 'yellow',
                'ERROR': 'red'
            },
            secondary_log_colors={},
            style='%'
        )
        self.logger = logging.getLogger('root')
        stream = logging.StreamHandler()
        stream.setFormatter(formatter)
        self.logger.addHandler(stream)
        logging.basicConfig(level=loglevel, format='%(levelname)s: %(message)s')
        

    def print(self, *args:tuple, inc:int=0, set_to_offset:Optional[int]=None, loglevel:int=logging.INFO, callerlog:bool=True):
        """
        Helper function to quickly disable output.
        Automatically adds current offset.
        """

        if callerlog:
            the_class = None
            the_method = '[none]'
            try:
                stack = inspect.stack()
                the_method = stack[1][0].f_code.co_name
                the_class = stack[1][0].f_locals["self"].__class__.__name__
            except BaseException as err:
                pass
            finally:
                if the_class:
                    caller = the_class + '.' + the_method
                else:
                    caller = the_method
            depth = len(inspect.stack(0)) - 4
        else:
            depth = 0
            caller = ''
        self.logger.log(loglevel, ' '*depth + f'[{self.offset:#010x}]/{caller}: %s', *args)

        self.offset += inc

        if isinstance(set_to_offset, int):
            self.offset = set_to_offset

        return self.offset

    def get_pos(self) -> int:
        """
        Returns current position.
        """
        return self.offset

    def set_pos(self, pos):
        """
        Sets current position.
        """
        self.offset = pos

    def set_debug_level(self):
        """
        Enables output.
        """
        self.loglevel = logging.DEBUG

    def disable(self):
        """
        Disables output.
        """
        self.loglevel = logging.ERROR

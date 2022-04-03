import builtins
import inspect
import logging
from typing import Callable

from computation_graph import composers


def _debug_with_frame(debugger):
    def debug(f):
        frame = inspect.currentframe().f_back

        def d(x):
            debugger(x, frame)
            return x

        return composers.compose_unary(d, f)

    return debug


def _debug_inner(x, frame):
    logging.info(
        f"Debug prompt for {frame.f_code.co_filename}:{frame.f_lineno}. Hit x+enter to see current value."
    )
    builtins.breakpoint()


#: Makes a pdb breakpoint with the node output (prints the line number!).
debug = _debug_with_frame(_debug_inner)


def _debug_log_inner(x, frame):
    logging.info(f"{frame.f_code.co_filename}:{frame.f_lineno} output:\n{x}")


#: Prints a debug log with the node output (with a line number!).
debug_log = _debug_with_frame(_debug_log_inner)


def name_callable(f: Callable, name: str) -> Callable:
    f.__name__ = name
    return f

import os
from distutils import util

DEBUG_SAVE_COMPUTATION_TRACE = bool(
    util.strtobool(os.getenv("CG_DEBUG_SAVE_COMPUTATION_TRACE", "false")),
)

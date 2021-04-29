# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module contains debugging helpers."""
import logging
import os

# Map for translating between COMPILER_GYM_DEBUG levels to python logging
# severity values.
_DEBUG_LEVEL_LOGGING_LEVEL_MAP = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
    3: logging.DEBUG,
}

_LOGGING_LEVEL_DEBUG_LEVEL_MAP = {
    v: k for k, v in _DEBUG_LEVEL_LOGGING_LEVEL_MAP.items()
}


def get_debug_level() -> int:
    """Get the debugging level.

    The debug level is a non-negative integer that controls the verbosity of
    logging messages and other debugging behavior. At each level, the types of
    messages that are logged are:

    * :code:`0` - only non-fatal errors are logged (default).
    * :code:`1` - extra warnings message are logged.
    * :code:`2` - enables purely informational logging messages.
    * :code:`3` and above - extremely verbose logging messages are enabled that
      may be useful for debugging.

    The debugging level can be set using the :code:`$COMPILER_GYM_DEBUG`
    environment variable, or by calling :func:`set_debug_level`.

    :return: A non-negative integer.
    """
    return max(int(os.environ.get("COMPILER_GYM_DEBUG", "0")), 0)


def get_logging_level() -> int:
    """Returns the logging level.

    The logging level is not set directly, but as a result of setting the debug
    level using :func:`set_debug_level`.

    :return: An integer.
    """
    return _DEBUG_LEVEL_LOGGING_LEVEL_MAP.get(get_debug_level(), logging.DEBUG)


def set_debug_level(level: int) -> None:
    """Set a new debugging level.

    See :func:`get_debug_level` for a description of the debug levels.

    The debugging level should be set first when interacting with CompilerGym as
    many CompilerGym objects will check the debug level only at initialization
    time and not throughout their lifetime.

    Setting the debug level affects the entire process and is not thread safe.
    For granular control of logging information, consider instead setting a
    :code:`logging.Logger` instance on :code:`CompilerEnv.logger`.

    :param level: The debugging level to use.
    """
    os.environ["COMPILER_GYM_DEBUG"] = str(level)

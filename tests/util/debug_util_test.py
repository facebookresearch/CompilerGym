# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for //compiler_gym/util:timer."""
import logging
import os

import pytest

from compiler_gym.util import debug_util as dbg
from tests.test_main import main


@pytest.fixture(scope="function")
def temporary_environ():
    old_env = os.environ.copy()
    os.environ.clear()
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_env)


def test_get_debug_level_environment_variable(temporary_environ):
    del temporary_environ
    os.environ["COMPILER_GYM_DEBUG"] = "0"
    assert dbg.get_debug_level() == 0
    os.environ["COMPILER_GYM_DEBUG"] = "1"
    assert dbg.get_debug_level() == 1


def test_get_and_set_debug_level(temporary_environ):
    del temporary_environ
    dbg.set_debug_level(0)
    assert dbg.get_debug_level() == 0
    dbg.set_debug_level(1)
    assert dbg.get_debug_level() == 1


def test_negative_debug_level(temporary_environ):
    del temporary_environ
    dbg.set_debug_level(-1)
    assert dbg.get_debug_level() == 0


def test_out_of_range_debug_level(temporary_environ):
    del temporary_environ
    dbg.set_debug_level(15)
    assert dbg.get_debug_level() == 15


def test_get_logging_level(temporary_environ):
    del temporary_environ
    dbg.set_debug_level(0)
    assert dbg.get_logging_level() == logging.ERROR
    dbg.set_debug_level(1)
    assert dbg.get_logging_level() == logging.WARNING
    dbg.set_debug_level(2)
    assert dbg.get_logging_level() == logging.INFO
    dbg.set_debug_level(3)
    assert dbg.get_logging_level() == logging.DEBUG
    dbg.set_debug_level(4)
    assert dbg.get_logging_level() == logging.DEBUG


if __name__ == "__main__":
    main()

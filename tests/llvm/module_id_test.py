# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests that module and source IDs are stripped in the LLVM modules."""
from compiler_gym.envs.llvm import LlvmEnv
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm", "tests.pytest_plugins.common"]


def test_no_module_id_builtin_benchmark(env: LlvmEnv):
    """Test that the module and source IDs are stripped in shipped benchmark."""
    env.reset("cBench-v1/crc32")
    ir = env.ir

    print(ir)  # For debugging in case of error.
    assert "; ModuleID = '-'\n" in ir
    assert '\nsource_filename = "-"\n' in ir


def test_no_module_id_custom_benchmark(env: LlvmEnv):
    """Test that the module and source IDs are stripped in custom benchmark."""

    with open("source.c", "w") as f:
        f.write("int A() {return 0;}")
    benchmark = env.make_benchmark("source.c")
    env.reset(benchmark=benchmark)
    ir = env.ir

    print(ir)  # For debugging in case of error.
    assert "; ModuleID = '-'\n" in ir
    assert '\nsource_filename = "-"\n' in ir


if __name__ == "__main__":
    main()

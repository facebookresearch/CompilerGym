#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from argparse import ArgumentParser


def make_config(argv):
    parser = ArgumentParser()
    parser.add_argument(
        "--out-file-path", type=str, required=True, help="Path to the generated config."
    )
    parser.add_argument("--enable-llvm-env", action="store_true")
    parser.add_argument("--enable-mlir-env", action="store_true")
    args = parser.parse_args(args=argv[1:])
    with open(args.out_file_path, "w") as f:
        f.write(f"enable_llvm_env = {args.enable_llvm_env}\n")
        f.write(f"enable_mlir_env = {args.enable_mlir_env}\n")


if __name__ == "__main__":
    make_config(sys.argv)

#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A script to auto-populate RST files from the CompilerGym header files.

Usage:

    $ python generate_cc_rst.py
"""
import os
from pathlib import Path
from typing import List

SOURCES = Path("../compiler_gym")
OUTPUT_DIR = Path("source/cc")


def header(message, underline="="):
    underline = underline * (len(str(message)) // len(underline))
    return f"{message}\n{underline}"


def main():
    valid_files: List[Path] = []
    for root, _, files in os.walk(SOURCES):
        if "third_party" in root:
            continue
        headers = [
            f
            for f in files
            if (f.endswith(".h") or f.endswith(".proto")) and not f.endswith("Impl.h")
        ]
        if not headers:
            continue

        while root.startswith("../"):
            root = root[len("../") :]
        root = Path(root)

        (OUTPUT_DIR / root).parent.mkdir(parents=True, exist_ok=True)
        output_path = Path(f"{OUTPUT_DIR / root}.rst")
        valid_files.append(output_path)
        print("Generating", output_path)
        with open(output_path, "w") as f:
            print(header(str(root)), file=f)
            print(file=f)
            print(".. contents::", file=f)
            print("   :local:", file=f)
            for header_name in headers:
                print(file=f)
                print(header(header_name, "-"), file=f)
                print(file=f)
                print(f':code:`#include "{root}/{header_name}"`', file=f)
                print(file=f)
                print(f".. doxygenfile:: {root}/{header_name}", file=f)

    for root, _, files in os.walk(OUTPUT_DIR):
        for file in files:
            path = Path(root) / file
            if path not in valid_files:
                print("rm", path)
                path.unlink()


if __name__ == "__main__":
    main()

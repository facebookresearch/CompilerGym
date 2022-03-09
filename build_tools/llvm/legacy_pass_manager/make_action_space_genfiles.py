# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Build generated files from a list of passes.

This script reads from stdin a list of passes and generates files so that these
passes can be used as an action space.

Usage:

    $ make_action_space_genfiles.py <output-directory> < <pass-list>

The following files are generated:

<outdir>/ActionHeaders.h
------------------------
    Example:

        #pragma once
        #include "llvm/LinkAllPasses.h"
        #include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
        ...

    This file includes the set of LLVM headers that must be included to use the
    passes.

<outdir>/ActionEnum.h
---------------------
    Example:

        enum class LlvmAction {
          ADD_DISCRIMINATORS_PASS,
          AGGRESSIVE_DCEPASS,
          ...
        }

    This defines an enum that names all of the passes.

<outdir>/ActionSwitch.h
-----------------------
    Example:

        #define HANDLE_ACTION(action, handlePass) \
          switch (action) {  \
            case LlvmAction::ADD_DISCRIMINATORS_PASS: \
              handlePass(llvm::createAddDiscriminatorsPass()); \
              break; \
            case LlvmAction::AGGRESSIVE_DCEPASS: \
              handlePass(llvm::createAggressiveDCEPass()); \
              break; \
          ...
        }

    To use the generated switch, call the HANDLE_ACTION() macro using an
    LlvmAction enum value and a handlePass function which accepts a pass
    instance as input.

<outdir>/flags.txt
-------------------------
    Example:

        -add-discriminators
        -adce
        ...

    A list of names for each pass.

<outdir>/flag_descriptions.txt
---------------------------------
    Example:

        Add DWARF path discriminators
        Aggressive Dead Code Elimination
        ...

    A list of descriptions of each pass.
"""
import csv
import logging
import sys
from pathlib import Path

from common import Pass
from config import LLVM_ACTION_INCLUDES

logger = logging.getLogger(__name__)


def process_pass(pass_, headers, enum_f, switch_f):
    """Extract and process transform passes in header."""
    if pass_.header:
        # Strip a leading "include/" from the header path.
        header = pass_.header
        if header.startswith("include/"):
            header = header[len("include/") :]
        headers.add(header)

    # The name of the pass in UPPER_PASCAL_CASE.
    enum_name = pass_.flag[1:].replace("-", "_").upper()
    print(f"  {enum_name},", file=enum_f)
    print(f"    case LlvmAction::{enum_name}: \\", file=switch_f)
    print(f"      handlePass(llvm::create{pass_.name}()); \\", file=switch_f)
    print("      break; \\", file=switch_f)


def make_action_sources(pass_iterator, outpath: Path):
    """Generate the enum and switch content."""
    total_passes = 0
    headers = set(LLVM_ACTION_INCLUDES)

    passes = sorted(list(pass_iterator), key=lambda p: p.name)

    switch_path = Path(outpath / "ActionSwitch.h")
    enum_path = Path(outpath / "ActionEnum.h")
    include_path = Path(outpath / "ActionHeaders.h")
    flags_path = Path(outpath / "flags.txt")
    descriptions_path = Path(outpath / "flag_descriptions.txt")

    with open(switch_path, "w", encoding="utf-8") as switch_f, open(
        enum_path, "w", encoding="utf-8"
    ) as enum_f:
        print("enum class LlvmAction {", file=enum_f)
        print("#define HANDLE_ACTION(action, handlePass) \\", file=switch_f)
        print("  switch (action) {  \\", file=switch_f)
        for pass_ in passes:
            total_passes += 1
            process_pass(pass_, headers, enum_f, switch_f)
        print("};", file=enum_f)
        print("  }", file=switch_f)

    logger.debug("Generated %s", switch_path.name)
    logger.debug("Generated %s", enum_path.name)

    with open(include_path, "w", encoding="utf-8") as f:
        print("#pragma once", file=f)
        for header in sorted(headers):
            print(f'#include "{header}"', file=f)

    logger.debug("Generated %s", include_path.name)

    with open(flags_path, "w", encoding="utf-8") as f:
        print("\n".join(p.flag for p in passes), file=f)
    logger.debug("Generated %s", flags_path.name)

    with open(descriptions_path, "w", encoding="utf-8") as f:
        print("\n".join(p.description for p in passes), file=f)
    logger.debug("Generated %s", descriptions_path.name)

    logger.debug("Created genfiles for %s pass actions", total_passes)


def main(argv):
    """Main entry point."""
    outpath = Path(argv[1])
    assert outpath.is_dir(), f"Output directory not found: {outpath}"

    reader = csv.reader(sys.stdin, delimiter=",", quotechar='"')
    next(reader)
    outpath = Path(outpath).absolute().resolve()
    pass_iterator = (Pass(*row) for row in reader)
    make_action_sources(pass_iterator, outpath)


if __name__ == "__main__":
    main(sys.argv)

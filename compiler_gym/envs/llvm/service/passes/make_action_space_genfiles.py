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
        #include "include/llvm/LinkAllPasses.h"
        #include "include/llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
        ...

    This file includes the set of LLVM headers that must be included to use the
    passes.

<outdir>/ActionEnum.h
---------------------
    Example:

        enum class LlvmAction {
          ADD_DISCRIMINATORS_PASS,
          AGGRESSIVE_DCEPASS,
        ... }

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
        ...  }

    To use the generated switch, call the HANDLE_ACTION() macro using an
    LlvmAction enum value and a handlePass function which accepts a pass
    instance as input.

<outdir>/actions_list.txt
-------------------------
    Example:

        AddDiscriminatorsPass
        AggressiveDcepass
        ...

    A list of names for each pass.

<outdir>/actions_flags.txt
--------------------------
    Example:

        -add-discriminators
        -adce
        ...

    A list of flags for each pass.

<outdir>/actions_descriptions.txt
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
from typing import List

from compiler_gym.envs.llvm.service.passes.common import Pass
from compiler_gym.envs.llvm.service.passes.config import EXTRA_LLVM_HEADERS


def camel_case_split(string: str) -> List[str]:
    """Split camelCase into a list of words.

    E.g. "thisIsCamelCaps" -> ["this", "Is, "Camel", "Caps"].
    """
    words = [[string[0]]]

    for c in string[1:]:
        if words[-1][-1].islower() and c.isupper():
            words.append(list(c))
        else:
            words[-1].append(c)

    return ["".join(word) for word in words]


def process_pass(pass_, headers, enum_f, switch_f):
    """Extract and process transform passes in header."""
    if pass_.header:
        headers.add(pass_.header)

    # The name of the pass in UPPER_PASCAL_CASE.
    enum_name = "_".join(camel_case_split(pass_.name)).upper()
    print(f"  {enum_name},", file=enum_f)
    print(f"    case LlvmAction::{enum_name}: \\", file=switch_f)
    print(f"      handlePass(llvm::create{pass_.name}()); \\", file=switch_f)
    print("      break; \\", file=switch_f)


def make_action_sources(pass_iterator, outpath: Path):
    """Generate the enum and switch content."""
    total_passes = 0
    headers = set(EXTRA_LLVM_HEADERS)

    passes = sorted(list(pass_iterator), key=lambda p: p.name)

    switch_path = Path(outpath / "ActionSwitch.h")
    enum_path = Path(outpath / "ActionEnum.h")
    include_path = Path(outpath / "ActionHeaders.h")
    actions_path = Path(outpath / "actions_list.txt")
    flags_path = Path(outpath / "actions_flags.txt")
    descriptions_path = Path(outpath / "actions_descriptions.txt")

    with open(switch_path, "w") as switch_f, open(enum_path, "w") as enum_f:
        print("enum class LlvmAction {", file=enum_f)
        print("#define HANDLE_ACTION(action, handlePass) \\", file=switch_f)
        print("  switch (action) {  \\", file=switch_f)
        for pass_ in passes:
            total_passes += 1
            process_pass(pass_, headers, enum_f, switch_f)
        print("};", file=enum_f)
        print("  }", file=switch_f)

    logging.debug("Generated %s", switch_path.name)
    logging.debug("Generated %s", enum_path.name)

    with open(include_path, "w") as f:
        print("#pragma once", file=f)
        for header in sorted(headers):
            print(f'#include "{header}"', file=f)
    logging.debug("Generated %s", include_path.name)

    with open(actions_path, "w") as f:
        print(
            "\n".join(
                "".join(x.capitalize() for x in camel_case_split(p.name))
                for p in passes
            ),
            file=f,
        )
    logging.debug("Generated %s", actions_path.name)

    with open(flags_path, "w") as f:
        print("\n".join(p.flag for p in passes), file=f)
    logging.debug("Generated %s", flags_path.name)

    with open(descriptions_path, "w") as f:
        print("\n".join(p.description for p in passes), file=f)
    logging.debug("Generated %s", descriptions_path.name)

    logging.debug("Created genfiles for %s pass actions", total_passes)


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

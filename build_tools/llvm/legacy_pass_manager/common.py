# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import NamedTuple, Optional


class Pass(NamedTuple):
    """The declaration of an LLVM pass."""

    # The name of the pass, e.g. "AddDiscriminatorsPass".
    name: str
    # The opt commandline flag which turns this pass on, e.g. "-add-discriminators".
    flag: str
    # The docstring for this pass, as reported by `opt -help`. E.g. "Add DWARF path discriminators".
    description: str
    # The path of the C++ file which defines this pass, relative to the LLVM source tree root.
    source: str
    # The path of the C++ header which declares this pass, relative to the LLVM source tree root.
    # If the header path could not be inferred, this is None.
    header: Optional[str]
    # Boolean flags set in INITIALIZE_PASS().
    cfg: bool
    is_analysis: bool

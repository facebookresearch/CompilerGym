# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Extract a list of passes form the LLVM source tree.

Usage:

    $ extract_passes_from_llvm_source_tree /path/to/llvm/source/root

Optionally accepts a list of specific files to examine:

    $ extract_passes_from_llvm_source_tree /path/to/llvm/source/root /path/to/llvm/source/file

Implementation notes
--------------------

This implements a not-very-good parser for the INITIALIZE_PASS() family of
macros, which are used in the LLVM sources to declare a pass using it's name,
flag, and docstring. Parsing known macros like this is fragile and likely to
break as the LLVM sources evolve. Currently only tested on LLVM 10.0.

A more robust solution would be to parse the C++ sources and extract all classes
which inherit from ModulePass etc.
"""
import codecs
import csv
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from compiler_gym.envs.llvm.service.passes.common import Pass
from compiler_gym.envs.llvm.service.passes.config import CREATE_PASS_NAME_MAP

# A regular expression to match the start of an invocation of one of the
# InitializePass helper macros.
INITIALIZE_PASS_RE = r"(INITIALIZE_PASS|INITIALIZE_PASS_BEGIN|INITIALIZE_PASS_WITH_OPTIONS|INITIALIZE_PASS_WITH_OPTIONS_BEGIN)\("
# A regular expression to match static const string definitions.
CONST_CHAR_RE = r'^\s*static\s+const\s+char(\s+(?P<name>[a-zA-Z_]+)\s*\[\s*\]|\s*\*\s*(?P<ptr_name>[a-zA-Z_]+))\s*=\s*(?P<value>".+")\s*;'


class ParseError(ValueError):
    def __init__(self, message: str, source: str, components: List[str]):
        self.message = message
        self.source = source
        self.components = components


def parse_initialize_pass(
    source_path: Path, header: Optional[str], input_source: str, defines: Dict[str, str]
) -> Iterable[Pass]:
    """A shitty parser for INITIALIZE_PASS() macro invocations.."""
    # Squish down to a single line.
    source = re.sub(r"\n\s*", " ", input_source, re.MULTILINE)
    # Contract multi-spaces to single space.
    source = re.sub(r",", ", ", source)
    source = re.sub(r"\s+", " ", source)
    source = re.sub(r"\(\s+", "(", source)
    source = re.sub(r"\)\s+", ")", source)

    # Strip the INITIALIZE_PASS(...) macro.
    match = re.match(rf"^\s*{INITIALIZE_PASS_RE}(?P<args>.+)\)", source)
    if not match:
        raise ParseError("Failed to match INITIALIZE_PASS regex", source, [])
    source = match.group("args")

    components = []
    start = 0
    in_quotes = False
    in_comment = False
    for i in range(len(source)):
        if (
            not in_comment
            and source[i] == "/"
            and i < len(source) - 1
            and source[i + 1] == "*"
        ):
            in_comment = True
        if (
            in_comment
            and source[i] == "*"
            and i < len(source) - 1
            and source[i + 1] == "/"
        ):
            in_comment = False
            start = i + 2
        if source[i] == '"':
            in_quotes = not in_quotes
        if not in_quotes and source[i] == ",":
            components.append(source[start:i].strip())
            start = i + 2
    components.append(source[start:].strip())
    if len(components) != 5:
        raise ParseError(
            f"Expected 5 components, found {len(components)}", source, components
        )

    pass_name, arg, name, cfg, analysis = components
    # Strip quotation marks in arg and name.
    if not arg:
        raise ParseError(f"Empty arg: `{arg}`", source, components)
    if not name:
        raise ParseError(f"Empty name: `{name}`", source, components)

    while arg in defines:
        arg = defines[arg]
    while name in defines:
        name = defines[name]

    if not (arg[0] == '"' and arg[-1] == '"'):
        raise ParseError(f"Could not interpret arg `{arg}`", source, components)
    arg = arg[1:-1]
    if not (name[0] == '"' and name[-1] == '"'):
        raise ParseError(f"Could not interpret name `{name}`", source, components)
    name = name[1:-1]

    # Convert cfg and analysis to bool.
    if cfg not in {"true", "false"}:
        raise ParseError(
            f"Could not interpret bool cfg argument `{cfg}`", source, components
        )
    if analysis not in {"true", "false"}:
        raise ParseError(
            f"Could not interpret bool analysis argument `{analysis}`",
            source,
            components,
        )
    cfg = cfg == "true"
    analysis = analysis == "true"

    opts = {
        "source": source_path,
        "header": header,
        "name": pass_name,
        "flag": f"-{arg}",
        "description": name,
        "cfg": cfg,
        "is_analysis": analysis,
    }

    pass_name_or_list = CREATE_PASS_NAME_MAP.get(pass_name, pass_name)

    if isinstance(pass_name_or_list, str):
        opts["name"] = pass_name_or_list
        yield Pass(**opts)
    else:
        for name in pass_name_or_list:
            opts["name"] = name
            yield Pass(**opts)


def build_defines(source: str) -> Dict[str, str]:
    """A quick-and-dirty technique to build a translation table from #defines
    and string literals to their values."""
    defines = {}
    lines = source.split("\n")
    for i in range(len(lines)):
        line = lines[i].strip()
        if line.startswith("#define"):
            # Match #define strings.
            components = line[len("#define ") :].split()
            name = components[0]
            value = " ".join(components[1:]).strip()
            if value == "\\":
                value = lines[i + 1].strip()
            defines[name] = value
        else:
            # Match string literals.
            match = re.match(CONST_CHAR_RE, line)
            if match:
                defines[match.group("name") or match.group("ptr_name")] = match.group(
                    "value"
                )
    return defines


def handle_file(source_path: Path) -> Tuple[Path, List[Pass]]:
    """Parse the passes declared in a file."""
    assert str(source_path).endswith(".cpp"), f"Unexpected file type: {source_path}"

    header = Path("include/llvm/" + str(source_path)[len("lib") : -len("cpp")] + "h")
    if not header.is_file():
        header = ""

    with codecs.open(source_path, "r", "utf-8") as f:
        source = f.read()

    defines = build_defines(source)

    passes: List[Pass] = []

    for match in re.finditer(INITIALIZE_PASS_RE, source):
        start = match.start()
        first_bracket = source.find("(", start)
        bracket_depth = 1
        end = first_bracket
        for end in range(first_bracket + 1, len(source)):
            if source[end] == "(":
                bracket_depth += 1
            elif source[end] == ")":
                bracket_depth -= 1
            if not bracket_depth:
                break

        try:
            passes += list(
                parse_initialize_pass(
                    source_path, header, source[start : end + 1], defines
                )
            )
        except ParseError as e:
            print(f"Parsing error: {e.message}", file=sys.stderr)
            print(f"Parsed components: {e.components}", file=sys.stderr)
            print(f"In line: {e.source}", file=sys.stderr)
            print(f"In file: {source_path}", file=sys.stderr)
            print("Fatal error. Aborting now.", file=sys.stderr)
            sys.exit(1)

    if passes:
        logging.debug(
            f"Extracted {len(passes)} {'passes' if len(passes) - 1 else 'pass'} from {source_path}",
        )
    else:
        logging.debug(f"Found no passes in {source_path}")

    return passes


def main(argv):
    root = Path(argv[1])
    assert root.is_dir(), f"Not a directory: {root}"
    os.chdir(root)

    if len(argv) > 2:
        paths = [Path(path) for path in argv[2:]]
    else:
        # Get the names of all files which contain a pass definition.
        matching_paths = []
        grep = subprocess.check_output(
            ["grep", "-l", "-E", rf"^\s*{INITIALIZE_PASS_RE}", "-R", "lib/"],
            universal_newlines=True,
        )
        matching_paths += grep.strip().split("\n")
        logging.debug("Processing %s files ...", len(matching_paths))
        paths = [Path(path) for path in matching_paths]

    # Build a list of pass entries.
    rows = []
    for path in sorted(paths):
        passes = handle_file(path)
        if passes:
            rows += passes

    writer = csv.writer(sys.stdout, delimiter=",", quotechar='"')
    writer.writerow(Pass._fields)
    writer.writerows(sorted(rows, key=lambda r: r.name))


if __name__ == "__main__":
    main(sys.argv)

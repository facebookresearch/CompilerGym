# NCC: Neural Code Comprehension
# https://github.com/spcl/ncc
# Copyright 2018 ETH Zurich
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
# following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
# disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
# disclaimer in the documentation and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote
# products derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
# INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ==============================================================================
# flake8: noqa
"""Preprocess LLVM IR code to XFG for inst2vec training"""
import copy
import os
import pickle
import re
from typing import Dict

import networkx as nx

from compiler_gym.third_party.inst2vec import rgx_utils as rgx


########################################################################################################################
# LLVM IR preprocessing
########################################################################################################################
def GetFunctionsDeclaredInFile(bytecode_lines):
    functions_declared_in_file = []

    # Loop on all lines in data
    for line in bytecode_lines:

        # Check whether it contains a function declaration
        if "declare" in line and not line.startswith("call void"):
            # Find the function name
            func = re.match(r"declare .*(" + rgx.func_name + r")", line)
            assert func is not None, "Could not match function name in " + line
            func = func.group(1)

            # Add it to the list
            functions_declared_in_file.append(func)

    return functions_declared_in_file


def get_functions_declared_in_files(data):
    """
    For each file, construct a list of names of the functions declared in the file, before the corresponding statements
    are removed by pre-processing. The list is used later on in the graph construction to identify the names of
    functions declared in this file.
    :param data: input data as a list of files where each file is a list of strings
    :return: functions_declared_in_files: list of lists of names of the functions declared in this file
    """
    return [GetFunctionsDeclaredInFile(file) for file in data]


def keep(line):
    """
    Determine whether a line of code is representative
    and should be kept in the data set or not.
    :param line: string representing the line of code to test
    :return: boolean: True if the line is to be kept,
                      False if the line is to be discarded
    """
    # Ignore empty lines.
    if line == "":
        return False

    # Ignore comment lines (except labels).
    if line[0] == ";" and not line[0:9] == "; <label>":
        return False

    if line[0] == "!" or line[0] == "\n":
        return False

    if (
        line.strip()[0] == "{"
        or line.strip()[0] == "}"
        or line.strip()[0] == "["
        or line.strip()[0] == "]"
    ):
        return False

    # Ignore empty lines (NOTE: possible dupe of `if line == ''` above?).
    if len(line) == 0:
        return False

    if "source_filename" in line:
        return False

    if "target triple" in line:
        return False

    if "target datalayout" in line:
        return False

    if "attributes" in line:
        return False

    if "module asm " in line:
        return False

    if "declare" in line:
        return False

    modif_line = re.sub(r"\".*\"", "", line)
    if re.match(rgx.global_id + r" = .*alias ", modif_line):
        return False

    if re.search("call void asm", line):
        return False

    match = re.search(r"\$.* = comdat any", line)
    if match:
        return False

    match = re.match(r"\s+;", line)
    if match:
        return False

    # If none of the above matched, keep the line.
    return True


def remove_non_representative_code(data):
    """
    Remove lines of code that aren't representative of LLVM-IR "language"
    and shouldn't be used for training the embeddings
    :param data: input data as a list of files where each file is a list of strings
    :return: input data with non-representative lines of code removed
    """
    for i in range(len(data)):
        data[i] = [line for line in data[i] if keep(line)]

    return data


def remove_leading_spaces(data):
    """
    Remove the leading spaces (indentation) of lines of code
    :param data: input data as a list of files, where each file is a list of strings
    :return: input data with leading spaces removed
    """
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = data[i][j].strip()

    return data


def remove_trailing_comments_and_metadata(data):
    """
    Remove comments, metadata and attribute groups trailing at the end of a line
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    for i in range(len(data)):
        for j in range(len(data[i])):

            line = data[i][j]

            # If the line contains a trailing metadata
            pos = line.find("!")
            if pos != -1:
                # Remove metadatas which are function arguments
                while re.search(r"\(.*metadata !.*\)", line) is not None:
                    line = re.sub(r"(, )?metadata !\d+(, )?", "", line)
                    line = re.sub(r"(, )?metadata !\w+(, )?", "", line)
                    line = re.sub(r"metadata !\d+(, )?", "", line)
                    line = re.sub(r"metadata !\w+(, )?", "", line)
                    pos = line.find("!")
            if pos != -1:
                # Check whether the '!' is part of a string expression
                pos_string = line[:pos].find('c"')
                if (
                    pos_string == -1
                ):  # there is no string expression earlier on the line
                    line = line[:pos].strip()  # erase from here to the end of the line
                    if line[-1] == ",":  # can happen with !tbaa
                        line = line[:-1].strip()
                else:  # there is a string expression earlier on the line
                    pos_endstring = line[pos_string + 2 : pos].find('"')
                    if (
                        pos_endstring != -1
                    ):  # the string has been terminated before the ;
                        line = line[
                            :pos
                        ].strip()  # erase from here to the end of the line
                        if line[-1] == ",":  # can happen with !tbaa
                            line = line[:-1].strip()

            # If the line contains a trailing attribute group
            pos = line.find("#")
            if pos != -1:
                # Check whether the ';' is part of a string expression
                s = re.search(r'c".*"', line[:pos])
                if not s:  # there is no string expression earlier on the line
                    line = line[:pos].strip()  # erase from here to the end of the line
                else:  # there is a string expression earlier on the line
                    pos_endstring = s.end()
                    if (
                        pos_endstring != -1
                    ):  # the string has been terminated before the ;
                        line = line[
                            :pos
                        ].strip()  # erase from here to the end of the line

            data[i][j] = line

    return data


def collapse_stmt_units_to_a_line(data):
    """
    Some statements are written on several lines though they really are just one statement
    Detect and collapse these
    :param data: input data as a list of files where each file is a list of strings
    :return: modified input data
    """
    to_track = ""
    erase_token = "to_erase"  # Helper variable to mark lines to be erased
    separator = "\n "

    # Detect multi-line statements and collapse them
    for file in data:
        for i in range(len(file)):

            if file[i] == to_track:
                print("Found", to_track)

            if re.match(rgx.local_id + " = landingpad", file[i]):
                if i + 1 < len(file):
                    if (
                        re.match(r"cleanup", file[i + 1])
                        or re.match(r"filter", file[i + 1])
                        or re.match(r"catch", file[i + 1])
                    ):
                        file[i] += separator + file[i + 1]  # collapse lines
                        file[i + 1] = erase_token  # mark as "to erase"
                    else:
                        continue
                if i + 2 < len(file):
                    if (
                        re.match(r"cleanup", file[i + 2])
                        or re.match(r"filter", file[i + 2])
                        or re.match(r"catch", file[i + 2])
                    ):
                        file[i] += separator + file[i + 2]  # collapse lines
                        file[i + 2] = erase_token  # mark as "to erase"
                    else:
                        continue
                if i + 3 < len(file):
                    if (
                        re.match(r"cleanup", file[i + 3])
                        or re.match(r"filter", file[i + 3])
                        or re.match(r"catch", file[i + 3])
                    ):
                        file[i] += separator + file[i + 3]  # collapse lines
                        file[i + 3] = erase_token  # mark as "to erase"
                    else:
                        continue
            elif re.match(r"switch", file[i]):
                for j in range(i + 1, len(file)):
                    if re.search(r"i\d+ -?\d+, label " + rgx.local_id, file[j]):
                        # if this statement is part of the switch,
                        file[i] += separator + file[j]  # collapse lines
                        file[j] = erase_token  # mark as "to erase"
                    else:
                        # if this statement isn't part of the switch
                        file[i] += "]"  # add closing bracket
                        break
            elif re.search(r"invoke", file[i]):
                if i + 1 < len(file):
                    if re.match(
                        r"to label " + rgx.local_id + " unwind label " + rgx.local_id,
                        file[i + 1],
                    ):
                        file[i] += separator + file[i + 1]  # collapse lines
                        file[i + 1] = erase_token  # mark as "to erase"

    # Erase statements which have been rendered superfluous from collapsing
    for i in range(len(data)):
        data[i] = [line for line in data[i] if line != erase_token]

    return data


def remove_structure_definitions(data):
    """
    Remove lines of code that aren't representative of LLVM-IR "language"
    and shouldn't be used for training the embeddings
    :param data: input data as a list of files where each file is a list of strings
    :return: input data with non-representative lines of code removed
    """
    for i in range(len(data)):
        data[i] = [
            line
            for line in data[i]
            if not re.match("%.* = type (<?{ .* }|opaque|{})", line)
        ]

    return data


def preprocess(data):
    """Pre-processing of source code:
    - remove non-representative lines of code
    - remove leading spaces (indentation)
    - remove trailing comments and metadata
    :param data: input data as a list of files where each file is a list of strings
    :return: preprocessed_data: modified input data
             functions_declared_in_files:
    """
    functions_declared_in_files = get_functions_declared_in_files(data)
    data = remove_non_representative_code(data)
    data = remove_leading_spaces(data)
    data = remove_trailing_comments_and_metadata(data)
    data = collapse_stmt_units_to_a_line(data)
    preprocessed_data = copy.deepcopy(data)
    preprocessed_data = remove_structure_definitions(preprocessed_data)

    return preprocessed_data, functions_declared_in_files


########################################################################################################################
# XFG-transforming (inline and abstract statements)
########################################################################################################################
# Helper regexs for structure type inlining
vector_type = "<\d+ x " + rgx.first_class_type + ">"
array_type = "\[\d+ x " + rgx.first_class_type + "\]"
array_of_array_type = "\[\d+ x " + "\[\d+ x " + rgx.first_class_type + "\]" + "\]"
function_type = (
    rgx.first_class_type
    + " \("
    + rgx.any_of([rgx.first_class_type, vector_type, array_type, "..."], ",")
    + "*"
    + rgx.any_of([rgx.first_class_type, vector_type, array_type, "..."])
    + "\)\**"
)
structure_entry = rgx.any_of(
    [
        rgx.first_class_type,
        vector_type,
        array_type,
        array_of_array_type,
        function_type,
    ]
)
structure_entry_with_comma = rgx.any_of(
    [
        rgx.first_class_type,
        vector_type,
        array_type,
        array_of_array_type,
        function_type,
    ],
    ",",
)
literal_structure = (
    "(<?{ " + structure_entry_with_comma + "*" + structure_entry + " }>?|opaque|{})"
)
literal_structure_with_comma = literal_structure + ", "


def construct_struct_types_dictionary_for_file(data):
    """
    Construct a dictionary of structure names
    :param data: list of strings representing the content of one file
    :return: data: modified input data
             ready: dictionary of structure names
    """
    # Optional: tracking
    to_track = ""

    # Three dictionaries
    to_process = dict()  # contains non-literal structures
    to_inline = dict()  # contains literal structures to be inlined in "to_process"
    ready = dict()  # contains literal structures which have already been inlined

    # Helper strings
    struct_prev = [structure_entry, literal_structure]
    struct_prev_with_comma = [
        structure_entry_with_comma,
        literal_structure_with_comma,
    ]
    use_previously_inlined_stmts = False

    # Put all "type" expressions from "data" into "to_process"
    for stmt in data:
        if len(to_track) > 0:
            if to_track in stmt:
                print("Found statement ", to_track)
        if re.match(rgx.struct_name + r" = type <?{?", stmt):
            k = re.sub(r"(" + rgx.struct_name + r") = type <?{?.*}?>?$", r"\g<1>", stmt)
            v = re.sub(rgx.struct_name + " = type (<?{?.*}?>?)$", r"\g<1>", stmt)
            to_process[k] = v

    # Loop over contents of "to_process"
    for i in list(to_process.items()):
        # Move the literal structures to to_inline
        if re.match(literal_structure, i[1]):
            to_inline[i[0]] = i[1]
            del to_process[i[0]]

    # Helper variables for iteration checks
    counter = 0
    prev_to_process_len = len(to_process)

    # While "to_process" is not empty
    while len(to_process) > 0:

        # Loop over contents of to_inline
        for i in list(to_inline.items()):
            # and inline these statements in to_process
            for p in list(to_process.items()):
                pattern = re.escape(i[0]) + rgx.struct_lookahead
                if re.search(pattern, p[1]):
                    to_process[p[0]] = re.sub(pattern, i[1], p[1])

        # Under certain circumstances
        if use_previously_inlined_stmts:
            # print("\t... using previously inlined statements")
            # Loop over contents of "to_process"
            for p in list(to_process.items()):
                # and inline these statements with structures from "ready"
                for i in list(ready.items()):
                    pattern = re.escape(i[0]) + rgx.struct_lookahead
                    if re.search(pattern, p[1]):
                        print("bingo")
                        to_process[p[0]] = re.sub(pattern, i[1], p[1])

        # Move contents of to_inline to ready
        ready.update(to_inline)
        to_inline = {}

        # Update possible structure entries
        if counter < 3:
            comp_structure_entry = rgx.any_of(struct_prev)
            comp_structure_entry_with_comma = rgx.any_of(struct_prev_with_comma)
            comp_structure = (
                "<?{ "
                + comp_structure_entry_with_comma
                + "*"
                + comp_structure_entry
                + " }>?"
            )
            struct_prev.append(comp_structure)
            struct_prev_with_comma.append(comp_structure + ", ")
        else:
            comp_structure = "<?{ [ <>{}\dx\[\]\(\)\.,\*%IDvfloatdubeipqcy]+}>?$"

        # Loop over contents of to_process
        for i in list(to_process.items()):
            if re.match(comp_structure, i[1]):
                to_inline[i[0]] = i[1]
                del to_process[i[0]]

        # Update counter
        counter += 1

        # Make sure progress as been made since the last iteration
        if len(to_process) == prev_to_process_len and counter > 3:
            # If this isn't the case, check if there is a type defined cyclically
            cycle_found = False
            for i in list(to_process.items()):
                # - Recursive, eg %intlist = type { %intlist*, i32 }
                # tracking
                if len(to_track) > 0:
                    if to_track in i[0]:
                        print("Found", to_track)
                if re.search(re.escape(i[0]) + rgx.struct_lookahead, i[1]):
                    cycle_found = True
                    new_entry = i[0] + "_cyclic"
                    to_inline[new_entry] = "opaque"
                    to_process[i[0]] = re.sub(
                        re.escape(i[0]) + rgx.struct_lookahead, new_entry, i[1]
                    )
                #                    break
                if not cycle_found:
                    # - Cyclic, eg
                    # %"class.std::ios_base": { i32 (...)**, i64, i32, %"struct.std::ios_base::_Callback_list"*, ...}
                    # %"struct.std::ios_base::_Callback_list": { opaque*, void (i32, %"class.std::ios_base"*, i32)* }
                    for j in list(to_process.items()):
                        if i != j and re.search(
                            re.escape(i[0]) + rgx.struct_lookahead, j[1]
                        ):
                            cycle_found = True
                            new_entry = i[0] + "_cyclic"
                            to_inline[new_entry] = "opaque"
                            to_process[j[0]] = re.sub(
                                re.escape(i[0]) + rgx.struct_lookahead, new_entry, j[1]
                            )
            #                            break

            # If no cyclic type definition was found although no progress was made since the last pass, abort
            if not cycle_found:
                if not use_previously_inlined_stmts:
                    # Perhaps some stmts which should be inlined are hiding in "ready": use these at the next pass
                    use_previously_inlined_stmts = True
                else:
                    assert cycle_found, (
                        "Counter step: "
                        + str(counter)
                        + ", could not inline "
                        + str(len(to_process))
                        + " statements : \n"
                        + string_of_items(to_process)
                    )
            else:
                use_previously_inlined_stmts = False  # reset

        prev_to_process_len = len(to_process)

        # Stopping condition in case we've been looping for a suspiciously long time
        assert counter < 1000, (
            "Could not inline "
            + str(len(to_process))
            + " statements after "
            + str(counter)
            + " steps: \n"
            + string_of_items(to_process)
        )

    # Move contents of "to_inline" to "ready"
    ready.update(to_inline)

    return data, ready


def GetStructTypes(ir: str) -> Dict[str, str]:
    """Extract a dictionary of struct definitions from the given IR.

    :param ir: A string of LLVM IR.
    :return: A dictionary of <name, def> entries, where <name> is the name of a struct
      definition (e.g. "%struct.foo"), and <def> is the definition of the member
      types, e.g. "{ i32 }".
    """
    try:
        _, dict_temp = construct_struct_types_dictionary_for_file(ir.split("\n"))
        return dict_temp
    except AssertionError as e:
        raise ValueError(e) from e


def PreprocessStatement(stmt: str) -> str:
    # Remove local identifiers
    stmt = re.sub(rgx.local_id, "<%ID>", stmt)
    # Global identifiers
    stmt = re.sub(rgx.global_id, "<@ID>", stmt)
    # Remove labels
    if re.match(r"; <label>:\d+:?(\s+; preds = )?", stmt):
        stmt = re.sub(r":\d+", ":<LABEL>", stmt)
        stmt = re.sub("<%ID>", "<LABEL>", stmt)
    elif re.match(rgx.local_id_no_perc + r":(\s+; preds = )?", stmt):
        stmt = re.sub(rgx.local_id_no_perc + ":", "<LABEL>:", stmt)
        stmt = re.sub("<%ID>", "<LABEL>", stmt)
    if "; preds = " in stmt:
        s = stmt.split("  ")
        if s[-1][0] == " ":
            stmt = s[0] + s[-1]
        else:
            stmt = s[0] + " " + s[-1]

    # Remove floating point values
    stmt = re.sub(rgx.immediate_value_float_hexa, "<FLOAT>", stmt)
    stmt = re.sub(rgx.immediate_value_float_sci, "<FLOAT>", stmt)

    # Remove integer values
    if (
        re.match("<%ID> = extractelement", stmt) is None
        and re.match("<%ID> = extractvalue", stmt) is None
        and re.match("<%ID> = insertelement", stmt) is None
        and re.match("<%ID> = insertvalue", stmt) is None
    ):
        stmt = re.sub(r"(?<!align)(?<!\[) " + rgx.immediate_value_int, " <INT>", stmt)

    # Remove string values
    stmt = re.sub(rgx.immediate_value_string, " <STRING>", stmt)

    # Remove index types
    if (
        re.match("<%ID> = extractelement", stmt) is not None
        or re.match("<%ID> = insertelement", stmt) is not None
    ):
        stmt = re.sub(r"i\d+ ", "<TYP> ", stmt)

    return stmt

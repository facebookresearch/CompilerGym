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
"""Preprocess LLVM IR code to XFG for inst2vec training"""
import copy
import os
import pickle
import re
from typing import Dict

import networkx as nx

from compiler_gym.third_party.inst2vec import inst2vec_utils as i2v_utils
from compiler_gym.third_party.inst2vec import rgx_utils as rgx


########################################################################################################################
# Helper functions: list and stmt handling
########################################################################################################################
def string_of_items(dic):
    """
    Return a string containing all keys of a dictionary, separated by a comma
    (Helper function for structure inlining)
    :param dic: dictionary [key=string: value=string]
    :return: string constructed of the dictionaries' keys
    """
    s = ""
    for k, v in dic.items():
        s += k + ": " + v + "\n"
    return s


def collapse_into_one_list(data):
    """
    Collapse list of list of strings into one list of strings
    :param data: list of list of strings
    :return: list of strings
    """
    data_ = list()
    for i in range(len(data)):
        for j in range(len(data[i])):
            data_.append(data[i][j])

    return data_


def string_from_list(l):
    """
    Construct a string from a list of strings
    :param l: list of strings
    :return: string containing elements of list l separated by a comma
    """
    s = l[0]
    if len(l) > 1:
        for i in range(len(l) - 1):
            # only add this string to the list if it is different from the previous strings
            e = l[i + 1]
            if e not in l[0 : i + 1]:
                s += ",\t\t" + e
    return s


def create_list_stmts(list_graphs):
    """
    Create a unique list of statements (strings) from a list of graphs in which statements are attributes of edges
    :param list_graphs: list of context-graphs (nodes = ids, edges = statements)
    :return: list_stmts: a unique list of statements (strings)
    """
    list_stmts = list()
    for G in list_graphs:
        edges_list = [e[2]["stmt"] for e in G.edges(data=True)]
        list_stmts += edges_list

    return list_stmts


########################################################################################################################
# Counting and statistics
########################################################################################################################
def get_stmt_counts(data_set, data_list):
    """
    Get statement counts
    :param data_set: set containing the elements from data_list but without repetitions and ordered
    :param data_list: list of string statements with repetitions and no ordering
    :return: data_count: dictionary with pairs [stmt, number of occurrences in data_list]
                         the order of the statements is the same as the one in data_set
             data_operations_count: list of tuples
                                    [string "tag level 1", "tag level 2", "tag level 3", int "number of occurrences"]
    """
    # Setup variables
    data_count = {x: 0 for x in data_set}
    data_operations_count = list()

    # Compute stmt counts (overall)
    print("Counting statement occurrences (overall)...")
    for stmt in data_list:
        data_count[stmt] += 1

    # Check that all stmts have been counted (for debugging purposes)
    total_stmt_count = sum(data_count.values())
    assert total_stmt_count == len(data_list), "Not all statements have been counted"

    # Compute stmt counts (by family)
    print("Counting statement occurrences (by family) ...")
    total_stmt_count = 0
    stmts_categorized = list()

    # Loop over stmt families
    for fam in rgx.llvm_IR_stmt_families:
        op_count = 0

        # loop on all stmts in data
        for i in range(len(data_set)):
            # if the regular expression for the family matches
            if re.match(fam[3], data_set[i], re.MULTILINE):
                # add the corresponding number of occurrences to the counter
                op_count += data_count[data_set[i]]
                stmts_categorized.append(i)

        # append the count to the list of number of occurrences
        data_operations_count.append([fam[0], fam[1], fam[2], op_count])

        # increase the total stmt count
        total_stmt_count += op_count

    # Check that all stmts have been categorized once and only once (debugging purposes)
    print("Starting categorization check ...")
    stmts_categorized = sorted(stmts_categorized)
    if stmts_categorized != list(range(len(data_set))):
        print("Tracking down the errors in categorization ... : ")
        for i in range(len(data_set)):
            num = stmts_categorized.count(i)
            if num == 0:
                print(data_set[i], "\n\tappears 0 times")
            if num > 1:
                print(data_set[i], "\n\tappears ", num, " times")

    assert stmts_categorized <= list(
        range(len(data_set))
    ), "Not all statements have been categorized"
    assert stmts_categorized >= list(
        range(len(data_set))
    ), "Some statements have been categorized multiple times"
    assert total_stmt_count == len(data_list), "Not all statements have been counted"

    return data_count, data_operations_count


def data_statistics(data, descr):
    """
    Compute and print some statistics on the data
    :param data: list of lists of statements (strings)
    :param descr: string description of the current step of the pipeline to add to output
    :return: source_data_list: list of statements
             source_data sorted set of statements
    """
    # Create a list of statements (strings) collecting the statements from all files
    source_data_list = collapse_into_one_list(data)

    # Create a sorted set of statements appearing in our data set
    source_data = sorted(set(source_data_list))

    # Get number of lines and the vocabulary size
    number_lines = len(source_data_list)
    vocabulary_size = len(source_data)

    # Construct output
    out = (
        "After "
        + descr
        + ":\n"
        + "--- {:<26}: {:>12,d}\n".format("Number of lines", number_lines)
        + "--- {:<26}: {:>12,d}\n".format("Vocabulary size", vocabulary_size)
    )
    print(out)

    # Return
    return source_data_list, source_data


########################################################################################################################
# Reading, writing and dumping files
########################################################################################################################


def read_data_files_from_folder(foldername):
    """
    Read all source files in folder
    Return a list of file contents, whereby each file content is a list of strings, each string representing a line
    :param foldername: name of the folder in which the data files to be read are located
    :return: a list of files where each file is a list of strings
    """
    # Helper variables
    data = list()
    file_names = list()
    file_count = 0

    print("Reading data from all files in folder ", foldername)
    listing = os.listdir(foldername + "/")
    to_subtract = file_count

    # Loop over files in folder
    for file in listing:
        if file[0] != "." and file[-3:] == ".ll":
            # If this isn't a hidden file and it is an LLVM IR file ('.ll' extension),
            # open file and import content
            f = open(os.path.join(foldername, file), "r")
            data.append(
                f.read().splitlines()
            )  # add this file as an element to the list "data"
            f.close()

            # Add file name to dictionary
            file_names.append(file)

            # Increment counters
            file_count += 1

    print("Number of files read from", foldername, ": ", file_count - to_subtract)
    print("Total number of files read for dataset", foldername, ": ", file_count)
    return data, file_names


def print_preprocessed_data(raw_data, foldername, filenames):
    """
    Write pre-processed code to file for future reference
    :param raw_data: a list of files where each file is a list of strings
    :param foldername: folder in which to print
    :param filenames: list of base file names
    :return:
    """
    # Make sure the directory exists - if not, create it
    foldername = os.path.join(foldername, "preprocessed")
    if not os.path.exists(foldername):
        os.makedirs(foldername)

    # Write pre-processed code to files
    i = 0
    for file in raw_data:
        filename = os.path.join(foldername, filenames[i][:-3] + "_preprocessed.txt")
        print("Writing pre-processed data to file ", filename)
        with open(filename, "w") as f:
            for l in file:
                f.write(l + "\n")
        i += 1


def print_data(data, filename):
    """
    Write pre-processed code to file for future reference
    :param data: a list of strings
    :param filename: name of file to print this to (string)
    :return:
    """
    print("Write data to file ", filename)
    with open(filename, "w") as f:
        for l in data:
            f.write(l + "\n")


def sort_key(x):
    """
    Helper function to sort nodes
    :param x: node
    :return: node name, node id type
    """
    id_part = x[0][1:]

    if id_part.isdigit():
        return x[0][0], int(x[0][1:])
    else:
        return x[0][0], 1


def print_node_family_to_file(G, f, nodetype):
    """
    Helper function for function "print_graph_to_file"
    :param G: graph
    :param f: file handle
    :param nodetype: string corresponding to the "id" of the node family to be printed
    """

    # Construct node family
    if nodetype == "root":
        node_family = [
            n for n in G.nodes() if G.out_degree(n) > 0 and G.in_degree(n) == 0
        ]
        node_family = sorted(node_family, key=sort_key)
    elif nodetype == "leaf":
        node_family = [
            n for n in G.nodes() if G.out_degree(n) == 0 and G.in_degree(n) >= 1
        ]
        node_family = sorted(node_family, key=sort_key)
    elif nodetype == "isolated":
        node_family = [n for n in G.nodes() if G.degree(n) == 0]
        node_family = sorted(node_family, key=sort_key)
    else:
        node_family = [
            n[0]
            for n in sorted(list(G.nodes(data=True)), key=sort_key)
            if n[1]["id"] == nodetype
        ]

    # Write to file
    f.write("#nodes: " + str(len(node_family)) + "\n")
    f.write("-" * 80 + "\n")
    for n in node_family:
        f.write("{n:<60}\n".format(n=n))


def print_graph_to_file(G, multi_edge_dic, folder, filename):
    """
    Print information about a graph to a file
    :param G: graph
    :param multi_edge_dic: dictionary of multi-edges
                           = edges for which a parallel edge connecting the same two end-nodes exists
    :param folder: folder in which to write
    :param filename: base name of the graph
    """
    # Print to file
    graph_filename = os.path.join(folder, filename[:-3] + ".txt")
    print("Printing graph to  file : ", graph_filename)

    with open(graph_filename, "w") as f:

        # GENERAL
        f.write("#nodes: " + str(G.number_of_nodes()) + "\n")
        f.write("#edges: " + str(G.number_of_edges()) + "\n\n")

        # INFORMATION ON NODES
        # all
        f.write("Nodes (" + str(G.number_of_nodes()) + "):\n")
        f.write("-" * 80 + "\n")
        for n, data in sorted(G.nodes(data=True), key=sort_key):
            f.write("{n:<60}, {w}\n".format(n=n[:60], w=data["id"]))

        # local
        f.write("\nLocal identifier nodes: \n")
        print_node_family_to_file(G, f, "local")

        # block references
        f.write("\nBlock reference nodes: \n")
        print_node_family_to_file(G, f, "label")

        # global
        f.write("\nGlobal nodes: \n")
        print_node_family_to_file(G, f, "global")

        # immediate value
        f.write("\nImmediate value nodes: \n")
        print_node_family_to_file(G, f, "imm_val")

        # ad_hoc
        f.write("\nAd hoc value nodes: \n")
        print_node_family_to_file(G, f, "ad_hoc")

        # leaf
        f.write("\nLeaf nodes: \n")
        print_node_family_to_file(G, f, "leaf")

        # root
        f.write("\nRoot nodes: \n")
        print_node_family_to_file(G, f, "root")

        # isolated
        f.write("\nIsolated nodes: \n")
        print_node_family_to_file(G, f, "isolated")
        f.write("\n\n")

        # INFORMATION ON EDGES
        # all
        f.write("Edges (" + str(G.number_of_edges()) + ")\n")
        f.write("-" * 80 + "\n")
        for a, b, data in sorted(G.edges(data=True), key=sort_key):
            f.write(
                "({a:<30}, {b:<30}) {w}\n".format(a=a[:30], b=b[:30], w=data["stmt"])
            )

        # data flow edges
        dataedges = [
            (str(n[0]), str(n[1]), str(n[2]))
            for n in sorted(list(G.edges(data=True)), key=sort_key)
            if n[2]["flow"] == "data"
        ]
        f.write("\nData flow edges: \n")
        f.write(
            "#edges: "
            + str(len(dataedges))
            + " ("
            + str(int(len(dataedges)) / G.number_of_edges() * 100)[:5]
            + "%)\n"
        )
        f.write("-" * 80 + "\n")
        for e in dataedges:
            f.write("({a:<30}, {b:<30}) {c}\n".format(a=e[0][:30], b=e[1][:30], c=e[2]))

        # control flow edges
        ctrledges = [
            (str(n[0]), str(n[1]), str(n[2]))
            for n in sorted(list(G.edges(data=True)), key=sort_key)
            if n[2]["flow"] == "ctrl"
        ]
        f.write("\nCtrl flow edges: \n")
        f.write(
            "#edges: "
            + str(len(ctrledges))
            + " ("
            + str(int(len(dataedges)) / G.number_of_edges() * 100)[:5]
            + "%)\n"
        )
        f.write("-" * 80 + "\n")
        for e in ctrledges:
            f.write("({a:<30}, {b:<30}) {c}\n".format(a=e[0][:30], b=e[1][:30], c=e[2]))

        # multi-edges
        f.write("\nMulti-edges: \n")
        multi_edge_list = list()
        for k, v in multi_edge_dic.items():  # Compile the multi-edges
            multi_edge_list += v
        f.write(
            "#multi-edges: "
            + str(len(multi_edge_list))
            + " ("
            + str(int(len(multi_edge_list)) / G.number_of_edges() * 100)[:5]
            + "%)\n"
        )
        f.write(
            "#node pairs connected by multi-edges: "
            + str(len(multi_edge_dic.keys()))
            + " ("
            + str(int(len(multi_edge_dic)) / G.number_of_edges() * 100)[:5]
            + "%)\n"
        )
        f.write("-" * 80 + "\n")
        for k, v_ in multi_edge_dic.items():
            n = re.match(r"(.*) \|\|\| (.*)", k)
            assert n is not None, "Could not identify nodes in " + k
            f.write("{m:<60} {p:<60}\n".format(m=n.group(1)[:60], p=n.group(2)[:60]))
            for v in v_:
                f.write("\t{}\n".format(v))
            f.write("\n")


def print_structure_dictionary(dic, folder, filename):
    """
    Print the dictionary of structures to a file
    :param dic: dictionary ["structure name", [list of possible values]]
    :param folder: name of folder in which to print dictionary
    :param filename: name of file in which to print dictionary
    :return:
    """
    # Print dictionary in alphabetical order
    dic_filename = os.path.join(folder, filename[:-3] + ".txt")
    print('Printing dictionary to file "', dic_filename)
    with open(dic_filename, "w") as f:
        f.write("{:<70}   {}\n\n".format("structure name", "literal value"))
        for key, value in sorted(dic.items()):
            f.write("{:<70}   {}\n".format(key, string_from_list(value)))


def PrintDualXfgToFile(D, folder, filename):
    """Print dual-XFG graph to file.

    :param D: dual-XFG graphs
    :param folder: name of folder in which to print dictionary
    :param filename: name of file in which to print dictionary
    """
    # Print to file
    graph_filename = os.path.join(folder, filename[:-3] + ".txt")
    print("Printing graph to  file : ", graph_filename)

    with open(graph_filename, "w") as f:
        # GENERAL
        f.write("#nodes: " + str(D.number_of_nodes()) + "\n")
        f.write("#edges: " + str(D.number_of_edges()) + "\n\n")

        # INFORMATION ON NODES
        f.write("Nodes (" + str(D.number_of_nodes()) + ")\n")
        f.write("-" * 80 + "\n")
        for n, _ in sorted(D.nodes(data=True), key=sort_key):
            f.write(f"{n:<60}\n")
        f.write("\n")
        # INFORMATION ON EDGES
        f.write("Edges (" + str(D.number_of_edges()) + ")\n")
        f.write("-" * 80 + "\n")
        for a, b, data in sorted(D.edges(data=True), key=sort_key):
            f.write(
                "({a:<37}, {b:<37}) {w}\n".format(a=a[:37], b=b[:37], w=data["weight"])
            )


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
# XFG-building
########################################################################################################################
def get_identifiers_from_line(line):
    """
    Extract identifiers (local, global and label) from a statement
    :param line: string: (part of) statement
    :return: lists of strings: m_loc, m_glob, m_label, m_label2
    """
    # Find label nodes
    m_label = m_label2 = list()
    if line.find("label") != -1 or re.match(rgx.local_id_no_perc + r":", line):
        m_label1 = re.findall("label (" + rgx.local_id + ")", line)
        if re.match(r"; <label>:" + rgx.local_id_no_perc + ":\s+", line):
            m_label2 = re.findall("<label>:(" + rgx.local_id_no_perc + "):", line)
        elif "invoke " in line:
            m_label2 = re.findall("label (" + rgx.local_id_no_perc + ")", line)
        else:
            m_label2 = re.findall(
                "<label>:(" + rgx.local_id_no_perc + ")", line
            ) + re.findall(r"(" + rgx.local_id_no_perc + r"):", line)
        for i in range(len(m_label2)):
            # put the '%' back in
            m_label2[i] = "%" + m_label2[i]
        m_label = m_label1 + m_label2

    # Find local identifier nodes
    modif_line = re.sub(r"\"[^\s]*\"", "", line)
    m_loc = sorted(re.findall(rgx.local_id, modif_line))

    # Remove what is actually an aggregate type and not a local identifier
    if len(m_loc) > 0:
        to_remove = []
        for m in m_loc:
            if m + "*" in line:
                to_remove.append(m)
            if m[:2] == '%"':
                to_remove.append(m)
            if " = phi " + m in line:
                to_remove.append(m)
            if " x " + m in line:
                to_remove.append(m)
            if " alloca " + m in line:
                to_remove.append(m)
        if len(to_remove) > 0:
            m_loc = [m for m in m_loc if m not in to_remove]

    # Find global identifier nodes
    m_glob = sorted(re.findall(rgx.global_id, line))

    # Remove label nodes from local nodes (they overlap)
    if len(m_label) > 0:
        m_loc = sorted(list(set(m_loc) - set(m_label)))

    # Return
    return m_loc, m_glob, m_label, m_label2


def find_outer_most_last_parenthesis(line):
    """
    Find outer-most last parenthesis of a line statement
    Used to identify the argument list in a function call
    :param line: a string representing a statement
    :return: a string representing this parenthesis with its content
    """
    # Find last closing parenthesis
    # It might not be last position as in the following statement:
    # invoke void @_Z12(%"class.std::basic_string"* nonnull sret %source_str, %"class.std::basic_string"* nonnull %agg)
    # to label %invoke.cont.276 unwind label %lpad.275
    end = 0
    if line[-1] == ")":
        # this corresponds to most cases
        start = len(line) - 1
    else:
        # look for the last closing parenthesis
        start = line.rfind(")")
        end = start
        if start == "-1":
            assert True, "Could not find right-most closing parenthesis in\n" + line

    # Stack opening and closing parenthesis to find the correct one
    close_bracket_count = -1
    for pos in range(start - 1, 0, -1):
        char = line[pos]
        if char == ")":
            close_bracket_count -= 1
        elif char == "(":
            close_bracket_count += 1
        else:
            pass
        if close_bracket_count == 0:
            start = pos
            break
    if end == 0:
        return line[start:]
    else:
        return line[start : end + 1]


def get_num_args_func(line, func_name=None):
    """
    Get the number of arguments in a line containing a function
    :param line: LLVM IR line
    :param func_name: function name
    :return num_args: number of arguments
            arg_list: list of arguments
    """
    modif_line = re.sub(
        r"<[^<>]+>", "", line
    )  # commas in vectors/arrays should not be counted as argument-separators
    arg_list_ = find_outer_most_last_parenthesis(modif_line)  # get last parenthesis
    if arg_list_ is None:
        # Make sure that this is the case because the function has no arguments
        # and not because there was in error in regex matching
        check = re.match(rgx.func_call_pattern + r"\(\)", modif_line)
        assert check is not None, (
            "Could not match argument list in:\n" + line + "\nFunction:\n" + func_name
        )
        num_args = 0
        arg_list = ""
    elif arg_list_ == "()":
        # Make sure that this is the case because the function has no arguments
        # and not because there was in error in regex matching
        check = re.match(rgx.func_call_pattern + r"\(\)", modif_line)
        if check is None:
            check = re.search(r" asm (?:sideeffect )?(\".*\")\(\)", modif_line)
        if check is None:
            check = re.search(rgx.local_id + r"\(\)", modif_line)
        if check is None:
            okay = line[-2:] == "()"
            if not okay:
                check = None
            else:
                check = True
        assert check is not None, (
            "Could not match argument list in:\n" + line + "\nFunction:\n" + func_name
        )
        num_args = 0
        arg_list = ""
    else:
        arg_list = arg_list_[1:-1]
        arg_list = re.sub(r"<[^<>]+>", "", arg_list)
        arg_list_modif = re.sub(r"\([^\(\)]+\)", "", arg_list)
        arg_list_modif = re.sub(r"\([^\(\)]+\)", "", arg_list_modif)
        arg_list_modif = re.sub(r"\([^\(\)]+\)", "", arg_list_modif)
        arg_list_modif = re.sub(r"\([^\(\)]+\)", "", arg_list_modif)
        arg_list_modif = re.sub(r"\"[^\"]*\"", "", arg_list_modif)
        arg_list_modif = re.sub(r"{.*}", "", arg_list_modif)
        num_args = len(re.findall(",", arg_list_modif)) + 1

    return num_args, arg_list


def construct_function_dictionary(file):
    """
    Construct a dictionary of functions which will be used to aid the construction of the context-graph
    :param file: list of statements
    :return: dictionary of functions
             keys: names of functions which are defined (not just declared) in this file
                    if named_args == True
             values: list: [shortened function name, boolean:called?, corresponding return statement, number of arguments,
                            list of arg names]
                    if named_args == False
             values: list: [shortened function name, boolean:called?, corresponding return statement, number of arguments]
    """

    # For debugging
    to_track = ""
    functions_defined_in_file = dict()
    func_name = ""

    # Loop over lines in file
    for line in file:

        # For debugging
        # if len(to_track) > 0:
        #   if line == to_track or to_track in line:
        #     print('Found line', line)

        # If it's a function definition
        if re.match(r"define", line):

            # When the definition of a function is detected, get the name of the function
            func_name_ = re.match(r"define .* (" + rgx.func_name + ")", line)
            assert func_name_ is not None, "Could not match function name in " + line
            func_name = func_name_.group(1)[1:]  # drop the leading '@'
            m_loc, m_glob, m_label, m_label2 = get_identifiers_from_line(line)
            if len(m_loc) > 0:
                named_args = True
            else:
                named_args = False

            # We will store a shortened version of the function's name since they can get very long
            # Find how many times the shortened version appear in other function names recorded thus far
            name_str = str(list(functions_defined_in_file.keys()))
            name_occurrences = len(re.findall(func_name[:20], name_str))
            # construct the shortened name
            func_name_short = func_name[:20] + "_" + str(name_occurrences)

            # Construct its list of arguments
            modified_line = re.sub(
                r"<[^<>]*>", "", line
            )  # commas in vectors/arrays not counted as arg-separators
            arg_list_ = re.match(
                r"define .* " + rgx.func_name + "\((.*)\)", modified_line
            )
            if arg_list_ is None:
                # Make sure that this is the case because the function has no arguments
                # and not because there was in error in regex matching
                check = re.match(r"define .* " + rgx.func_name + "\(\)", modified_line)
                assert check is not None, (
                    "Could not match argument list in "
                    + line
                    + "\n"
                    + "Modified line: "
                    + modified_line
                    + "\n"
                    + "Function name (to check): "
                    + func_name
                )
                num_args = 0
            else:
                num_args, arg_list_ = get_num_args_func(line)
                if num_args > 0:
                    arg_list = list()
                    arg_list_ = re.sub(r"\([^\(\)]+\)", "", arg_list_)
                    arg_list_ = re.sub(r"\([^\(\)]+\)", "", arg_list_)
                    arg_list_ = re.sub(r"<[^<>]+>", "", arg_list_)
                    arg_list_ = re.sub(r"<[^<>]+>", "", arg_list_)
                    arg_list_ = re.sub(r"<[^<>]+>", "", arg_list_)
                    arg_list_ = re.sub(r"<[^<>]+>", "", arg_list_)
                    arg_list_ = re.sub(r"<[^<>]+>", "", arg_list_)
                    args_ = arg_list_.split(", ")

                    try:
                        if len(args_) != num_args:
                            print(
                                "Could not compute the right number of arguments in "
                                + line
                                + "\n(a) "
                                + str(len(args_))
                                + "\n(b) "
                                + str(num_args)
                                + "\nwith arg-list: "
                                + arg_list_
                            )
                            raise ValueError("FunctionNotSupported")
                        if named_args:
                            for a in range(num_args):
                                arg_ = re.match(
                                    r".*( " + rgx.local_id + r"|\.\.\.)$", args_[a]
                                )
                                if arg_ is None:
                                    # Sometimes (eg. rodinia/openmp_particle_filter.ll),
                                    # some functions have unnamed args even though most are named
                                    # Check whether that is the case
                                    if re.match(
                                        rgx.any_type_or_struct
                                        + r"( nocapture| readonly| readnone| dereferenceable)*",
                                        args_[a],
                                    ):
                                        arg_list.append("%" + str(a))
                                    else:
                                        arg_ = re.match(r"(\.\.\.)$", args_[a])
                                        assert arg_ is not None, (
                                            "Could not identify argument name in \n"
                                            + line
                                            + "\nargument is\n"
                                            + args_[a]
                                        )
                                else:
                                    if arg_.group(1) == r"...":
                                        arg_list.append("three_dots")
                                    else:
                                        arg_list.append(
                                            arg_.group(1)[1:]
                                        )  # drop initial space
                    except ValueError:
                        raise

            # Construct dictionary entry for this function
            called = False
            if func_name in functions_defined_in_file.keys():
                called = functions_defined_in_file[func_name][1]
            if named_args:
                functions_defined_in_file[func_name] = [
                    func_name_short,
                    called,
                    "no_return",
                    num_args,
                    arg_list,
                ]
            else:
                functions_defined_in_file[func_name] = [
                    func_name_short,
                    called,
                    "no_return",
                    num_args,
                ]

        # If it's a return statement
        elif re.match(r"ret .*", line):
            if func_name:
                # add the return statement to the dictionary
                functions_defined_in_file[func_name][2] = line
                func_name = ""

        # If it's a call to a function defined in this file
        elif (
            re.match("(" + rgx.local_id + " = )?(tail )?(call|invoke) ", line)
            and "asm" not in line
            and " @" in line
        ):
            # Get the function name
            function_name_ = re.search(
                r"(" + rgx.func_name + r")( to .*)?\(.*\)($|\n)", line
            )
            if function_name_ is None:
                function_name_ = re.search(
                    r"(" + rgx.local_id + r")( to .*)?\(.*\)($|\n)", line
                )
                assert function_name_ is not None, (
                    "Could not identify function name in statement:\n" + line
                )
            else:
                function_name = function_name_.group(1)[1:]
                # If it is in the list of defined functions, change its entry to "called"
                if function_name in functions_defined_in_file.keys():
                    functions_defined_in_file[function_name][1] = True
                else:
                    functions_defined_in_file[function_name] = [
                        "REMOVE",
                        True,
                        "REMOVE",
                        num_args,
                    ]

    # Reconstruct dictionary removing the calls that did not have defines
    functions_defined_in_file_DEF = dict()
    for k, v in functions_defined_in_file.items():
        if v[0] != "REMOVE":
            functions_defined_in_file_DEF[k] = v

    # Make sure all function names have a corresponding return identifier
    for k, v in functions_defined_in_file_DEF.items():
        if k != "main":
            if v[1] == "no_return":
                print("WARNING! Function", k, "has no corresponding return statement")

    return functions_defined_in_file_DEF


def all_edges(G, nbunch=None, data=False):
    """
    Get a list of all (both incoming and outgoing) edges of (Multi)DiGraph G
    :param G: (Multi)DiGraph
    :param nbunch: list of nodes whose adjacent edges we want to find
    :param data: boolean: return list with or without nodes data
    :return: corresponding list of all (both incoming and outgoing) edges (no duplicates in list)
    """
    if data:
        result = [
            (e[0], e[1], e[2]["stmt"]) for e in G.in_edges(nbunch=nbunch, data=data)
        ]
        result += [
            (e[0], e[1], e[2]["stmt"]) for e in G.out_edges(nbunch=nbunch, data=data)
        ]
        return list(set(result))
    else:  # data == False
        result = list(G.in_edges(nbunch=nbunch, data=data))
        result += list(G.out_edges(nbunch=nbunch, data=data))
        ret = list(set(result))
    return ret


def all_neighbors(G, n):
    """
    Get a list of all neighbor-nodes (both predecessors and successors) of (Multi)DiGraph G
    :param G: (Multi)DiGraph
    :param n: list of nodes whose neighbours we want to find
    :return: corresponding list of all neighbor-nodes (no duplicates in list)
    """
    result = list(G.predecessors(n))
    result += list(G.successors(n))
    return list(set(result))


def all_degrees(G, n):
    """
    Get the sum of the in and out degress of a node in a (Multi)DiGraph G
    :param G: (Multi)DiGraph
    :param n: node whose degree we want to find
    :return: overall degree of the node
    """
    return G.in_degree(n) + G.out_degree(n)


def basic_block_leaf(G, node, ids_in_basic_block):
    """
    Test whether the node is a leaf node of a basic block
    :param G: Graph
    :param node: node to test
    :param ids_in_basic_block: list of IDs of other nodes in the basic block
    :return: boolean
    """
    bb_leaf = True
    if G.out_degree(node) > 0:
        for n in G.successors(node):
            if n in ids_in_basic_block:
                if G.node[n]["id"] != "ad_hoc":
                    # Check whether the outgoing edge is a "store"
                    for e in G.out_edges(node, data=True):
                        if "store" not in e[2]["stmt"]:
                            bb_leaf = False
                            break
    return bb_leaf


def add_node(G, func_prefix, node, id, ids_in_basic_block):
    """
    Wrapper around "add node"
    :param G: Graph
    :param func_prefix: function prefix
    :param node: node to add
    :param id: id of the node to add
    :param ids_in_basic_block: list of ids in current basic blocks
    """
    # Debugging
    node_check = ""
    if len(node_check) > 0:
        if node_check in node or node_check == node:
            print("Found node", node)
    assert node is not None, "Node none"

    # Add node
    G.add_node(func_prefix + node, id=id)
    if ids_in_basic_block is not None:
        if node[0] == "%" and id != "label" and node not in ids_in_basic_block:
            ids_in_basic_block.append(func_prefix + node)


def add_edge(G, parent_prefix, parent_node, child_prefix, child_node, stmt, flow):
    """
    Wrapper around "add edge"
    :param G: Graph
    :param parent_prefix: prefix of parent node
    :param parent_node: parent node
    :param child_prefix: prefix of child node
    :param child_node: child node
    :param stmt: statement corresponding to the edge to add
    :param flow: type of flow of the edge to add
    :return:
    """
    # Assert
    assert len(stmt.strip()) > 0

    # Debugging
    stmt_check = ""
    if len(stmt_check) > 0:
        if stmt_check in stmt or stmt_check == stmt:
            print("Found stmt", stmt)
    assert parent_node != "undef", "Found undef parent-node at stmt:\n" + stmt
    assert child_node != "undef", "Found undef child-node at stmt:\n" + stmt

    # Assert that the nodes have been added to the graph prior to this
    parent_node_ = parent_prefix + parent_node
    nodes = list(G.nodes())
    if parent_node_ not in nodes:
        raise ValueError(
            "Node not added to graph:\n"
            + parent_node_
            + "\nFound while trying to add edge:\n"
            + stmt
        )
    child_node_ = child_prefix + child_node
    if child_node_ not in nodes:
        raise ValueError(
            "Node not added to graph:\n"
            + child_node_
            + "\nFound while trying to add edge:\n"
            + stmt
        )

    # Add edge
    G.add_edge(parent_node_, child_node_, stmt=stmt, flow=flow)


def add_edge_dummy(G, parent_prefix, parent_node, stmt, ad_hoc_count):
    """
    Wrapper around "add edge" to add to connect a node to an ad hoc dummy node
    :param G: Graph
    :param parent_prefix: prefix of parent node
    :param parent_node: parent node
    :param stmt: statement corresponding to the edge to add
    :param ad_hoc_count: count of ad hoc nodes
    :return: updated ad_hoc_count
    """
    # Debugging
    stmt_check = ""
    if len(stmt_check) > 0:
        if stmt_check in stmt:
            print("Found stmt", stmt)

    # Assert that the nodes have been added to the graph prior to this
    parent_node_ = parent_prefix + parent_node
    assert parent_node_ in list(G.nodes()), (
        "Node not added to graph:\n"
        + parent_node_
        + "\nFound while trying to add edge:\n"
        + stmt
    )
    ad_hoc_node = "ad_hoc_" + str(ad_hoc_count)
    G.add_node(ad_hoc_node, id="ad_hoc")
    G.add_edge(parent_node_, ad_hoc_node, stmt=stmt, flow="path")
    return ad_hoc_count + 1


def add_stmts_to_graph(G, file, functions_defined_in_file, functions_declared_in_file):
    """
    Add all statements from a file to a graph
    :param G: (Multi)Digraph
    :param file: list of strings constituting a file
    :param functions_defined_in_file: dictionary of functions
           keys: names of functions which are defined (not just declared) in this file
                if named_args == True
           values: list: [shortened function name, boolean:called?, corresponding return statement, number of args, list of arg names]
                if named_args == False
           values: list: [shortened function name, corresponding return statement, number of arguments]
    :param functions_declared_in_file: list of names of the functions declared in this file
    :return: completed graph
    """

    # Helper-variables
    lines_not_added_to_graph = (
        list()
    )  # lines which couldn't be added to the graph (debugging purposes)
    G.add_node(
        "@0", id="global"
    )  # add a global reference node to connect declaration of gloal variales to
    glob_ref = list(G.nodes)[0]  # handle to global reference node
    func_prefix = ""  # function prefix (for construction of identifier nodes)
    block_ref = ""  # block reference
    ids_in_basic_block = list()
    ad_hoc_count = 0  # count of "ad-hoc"-nodes
    functions_declared_in_file = set(functions_declared_in_file)
    func_block_refs = dict()
    stmt_check = ""

    # Loop over the lines in the LLVM IR file
    for i, line in enumerate(file):

        # Debugging
        if stmt_check:
            if line == stmt_check or stmt_check in line:
                print("\nFound statement in", line)

        # Adapt to dragon-egg generated code
        if '%"ssa point"' in line:
            line = line.replace('%"ssa point"', '%"ssa_point"')
        elif '%"alloca point"' in line:
            line = line.replace('%"alloca point"', '%"alloca_point"')
        elif '%"<retval>' in line:
            line = line.replace('%"<retval>', '%"retval')

        ################################################################################################################
        # Add nodes and edges according to statement characteristics

        ################################################################################################################
        # Declaration of a global variable
        if re.match(
            rgx.global_id + r" =" + rgx.linkage + r"* constant ", line
        ) or re.match(rgx.global_id + r" =" + rgx.linkage + r"* global ", line):
            # (globref) --[stmt]--> (global variable)
            globvar = re.match(r"(" + rgx.global_id + r") =", line).group(1)
            add_node(G, "", globvar, "global", ids_in_basic_block)
            add_edge(G, "", glob_ref, "", globvar, line, "path")

        ################################################################################################################
        # Function definition
        elif re.match(r"define .* " + rgx.func_name + "\(.*\)", line):
            # We are in the body of a new function
            # eg define i32 @main() local_unnamed_addr #0 {

            # update previous function and function prefix:
            func_name = re.match(r"define .* (" + rgx.func_name + ")\(.*\)", line)
            assert func_name is not None, "Could not match function name in " + line
            func_name_ = func_name.group(1)[1:]
            func_prefix = functions_defined_in_file[func_name_][0] + "_"

            # update the block reference and add a node corresponding to the block reference
            if re.match(rgx.start_basic_block, file[i + 1]):
                # check if the next line is a block ref, then let that be the block reference
                label = re.match(rgx.start_basic_block, file[i + 1]).group(1)
                if label[0] != "%":
                    label = "%" + label
                if label[-1] == ":":
                    label = label[:-1]
                block_ref = func_prefix + label
            else:
                num_args = functions_defined_in_file[func_name_][3]
                block_ref = func_prefix + "%" + str(num_args)

            # +(block reference)
            func_block_refs[func_prefix] = block_ref
            add_node(G, "", block_ref, "label", ids_in_basic_block)
            ids_in_basic_block = list()  # start afresh

            # (globref) --[..define..]--> (block reference)
            add_edge(G, "", glob_ref, "", block_ref, line, "path")

            # Get list of arguments
            if re.search(rgx.local_id + r"(?!\* )(?=([\s,\)]|$))", line) is not None:
                # then the arguments are explicitely named
                arg_nodes = functions_defined_in_file[func_name_][4]
            else:
                # then the arguments are referred to as %0, %1, etc. though this isn't explicitely stated
                num_args = functions_defined_in_file[func_name_][3]
                arg_nodes = ["%" + str(i) for i in range(num_args)]

            # (blockref) --[..define..]--> (arguments)
            for a in arg_nodes:
                add_node(G, func_prefix, a, "local", ids_in_basic_block)
                add_edge(G, "", block_ref, func_prefix, a, line, "path")

        ################################################################################################################
        # Label (i.e., a new basic block)
        elif re.match(rgx.start_basic_block, line):
            # eg ; <label>:11:                                     ; preds = %8
            # eg .lr.ph.i:
            assert block_ref, "Empty block reference at line:\n" + line
            assert func_prefix, "Empty function prefix at line:\n" + line
            if all_degrees(G, block_ref) == 0:
                G.remove_node(
                    block_ref
                )  # if the previous block reference has not been used, delete it

            # Update block reference
            label_ = re.match("(?:.*<label>:)?(" + rgx.local_id_no_perc + "):?", line)
            assert label_ is not None, "Could not identify label in:\n" + line
            label = label_.group(1)
            if label[0] != "%":
                label = "%" + label
            if label[-1] == ":":
                label = label[:-1]
            block_ref = func_prefix + label
            add_node(G, func_prefix, label, "label", ids_in_basic_block)

            # Empty the list of variables in basic block
            ids_in_basic_block = list()

        ################################################################################################################
        # Variable assignment (except function calls)
        elif re.match(rgx.local_id + r" = (?!(tail )?(call|invoke) )", line):

            # Detect the assignee and add its node
            assignee_ = re.match(r"(" + rgx.local_id + ") = ", line)
            assert assignee_ is not None, "Could not identify assignee in:\n" + line
            assignee = assignee_.group(1)
            add_node(G, func_prefix, assignee, "local", ids_in_basic_block)

            if not re.match(rgx.local_id + r" = phi ", line):

                # This is just a regular assignment operation (not a phi-statement)
                # Get the operands
                m_loc, m_glob, m_label, m_label2 = get_identifiers_from_line(
                    re.sub(r"{.*}", "", line)
                )
                if assignee in m_loc:
                    m_loc.remove(assignee)
                operands = list()
                if len(m_loc) > 0:
                    operands = m_loc
                if len(m_glob) > 0:
                    for mg in m_glob:
                        if mg not in functions_declared_in_file:
                            operands.append(mg)

                # Connect operands to assignee
                no_parent = True
                if len(operands) > 0:
                    for op in operands:
                        # if operand is in this basic block, then the statement has a parent
                        if op[0] != "@" and func_prefix + op in ids_in_basic_block:
                            no_parent = False
                        # (operand) --[stmt]--> (assignee)
                        if not re.match(rgx.global_id, op):
                            if re.match(rgx.local_id, op):
                                add_node(G, func_prefix, op, "local", None)
                                add_edge(
                                    G,
                                    func_prefix,
                                    op,
                                    func_prefix,
                                    assignee,
                                    line,
                                    "data",
                                )
                        else:
                            if op not in list(G.nodes()):
                                add_node(G, "", op, "global", None)
                                add_edge(G, "", glob_ref, "", op, line, "path")
                            add_edge(G, "", op, func_prefix, assignee, line, "data")

                # If the statement has no parent in the present basic block, connect the paths
                if no_parent:
                    # (block ref) --[stmt]--> (assignee)
                    add_edge(G, "", block_ref, func_prefix, assignee, line, "path")

            else:

                # This is a phi statement
                # (block ref) --[stmt]--> (assignee)
                add_edge(G, "", block_ref, func_prefix, assignee, line, "path")

                # get a list of pairs of arguments
                m_ = re.findall(
                    r"\[ (%?"
                    + rgx.local_id_no_perc
                    + r"|true|false|<.*>|getelementptr inbounds \([ \d\w\[\]\*\.@,]+\)|.*"
                    + rgx.global_id
                    + r".*), (%?"
                    + rgx.local_id_no_perc
                    + ") \],?",
                    line,
                )
                if len(m_) == 0:
                    m_ = re.findall(
                        r"\[ (inttoptr \("
                        + rgx.base_type
                        + r" "
                        + rgx.immediate_value_int
                        + r" to "
                        + rgx.base_type_or_struct_name
                        + r"\**\)|%?-?"
                        + rgx.local_id_no_perc
                        + r"|"
                        + rgx.immediate_value
                        + r"|"
                        + r"|<.*>), "
                        + r"(%?-?"
                        + rgx.local_id_no_perc
                        + ") \]",
                        line,
                    )
                assert len(m_) > 0, (
                    "Could not identify arguments in phi-statement: " + line
                )

                # Loop over the list of arguments
                for m in m_:
                    if (
                        m[0][0] == "%"
                    ):  # if it's from a variable, not an immediate value
                        # (val nodes) --[stmt]--> (assignee)
                        add_node(G, func_prefix, m[0], "local", ids_in_basic_block)
                        add_edge(
                            G, func_prefix, m[0], func_prefix, assignee, line, "data"
                        )

                    elif (
                        re.match(r".*" + rgx.global_id + r".*", m[0])
                        or m[0][:13] == "getelementptr"
                    ):
                        # if it is from a global id
                        # (val nodes) --[stmt]--> (assignee)
                        m_g = re.search(rgx.global_id, m[0]).group(0)
                        if (
                            m_g in functions_declared_in_file
                            or m_g[1:] in functions_defined_in_file.keys()
                        ):
                            add_node(G, "", m_g, "global", None)
                            add_edge(G, "", glob_ref, "", m_g, line, "path")
                        add_edge(G, "", m_g, func_prefix, assignee, line, "data")

        ################################################################################################################
        # Not an assignment:
        else:

            ############################################################################################################
            # store
            if re.match("store ", line):
                # eg store float %11, float* %6, align 4
                m = re.match(
                    r"store (?:volatile )?\{?[\"\:\%\.\,\_\*\d\s\w\<\>]+\}? ("
                    + rgx.immediate_or_local_id
                    + '|undef), \{?["\:\%\.\,\_\*\d\s\w\<\>]+\}?\* ('
                    + rgx.local_or_global_id
                    + "|null)",
                    line,
                )
                if m is None:
                    m = re.match(
                        r"store (?:volatile )?\{?[\"\:\%\.\,\_\*\d\s\w\<\>]+\}? ("
                        + rgx.global_id
                        + '), \{?["\:\%\.\,\_\*\d\s\w\<\>]+\}?\* ('
                        + rgx.local_or_global_id
                        + ")",
                        line,
                    )
                    if m is None:
                        line_modif = re.sub(r"\([^\(\)]+\)", "", line)
                        line_modif = re.sub(r"\{[^\{\}]+\}", "", line_modif)
                        m = re.match(
                            r"store (?:volatile )?.* ("
                            + rgx.local_or_global_id
                            + ")(?: to .*)?, .*\* ("
                            + rgx.local_or_global_id
                            + ")",
                            line_modif,
                        )
                        if m is None:
                            (
                                m_loc_,
                                m_glob_,
                                m_label_,
                                m_label2_,
                            ) = get_identifiers_from_line(line)
                            m_imm_ = re.search(
                                r"(?<!%)(?<!align )"
                                + rgx.immediate_value
                                + r"(?!( x ))",
                                line,
                            )
                            l = m_loc_ + m_glob_
                            if len(l) == 1 and m_imm_ is not None:
                                l.append(m_imm_.group(0))
                            assert len(l) >= 2, (
                                "Cannot identify operands in:\n"
                                + line
                                + "\nGot: "
                                + str(l)
                            )
                            pos = list()
                            for ll in l:
                                pos_ = line.find(ll)
                                assert pos_ != -1, "Cannot find " + ll + " in " + line
                                pos.append(pos_)
                            val_ = min(pos)
                            for i in range(len(pos)):
                                if pos[i] == val_:
                                    m1 = l[i]
                                    break
                            l.remove(m1)
                            m2_ = l
                            m = "dummy"
                        else:
                            m1 = m.group(1)
                            m2_ = m.group(2)
                    else:
                        m1 = m.group(1)
                        m2_ = m.group(2)
                else:
                    m1 = m.group(1)
                    m2_ = m.group(2)

                assert m is not None, "Cannot not identify operands in:\n" + line
                assert m1 is not None, "m1 is none, stmt:\n" + line
                assert m2_ is not None, "m2 is none, stmt:\n" + line
                if not isinstance(m2_, list):
                    m2_ = [m2_]
                for m2 in m2_:
                    if re.match(rgx.global_id, m1):
                        if (
                            m1 in functions_declared_in_file
                            or m1[1:] in functions_defined_in_file.keys()
                        ):
                            add_node(G, "", m1, "global", None)
                            add_edge(G, "", glob_ref, "", m1, line, "path")
                        if re.match(rgx.global_id, m2):
                            if (
                                m2 in functions_declared_in_file
                                or m2[1:] in functions_defined_in_file.keys()
                            ):
                                add_node(G, "", m2, "global", None)
                                add_edge(G, "", glob_ref, "", m2, line, "path")
                            add_edge(G, "", m1, "", m2, line, "data")
                        else:
                            add_node(G, func_prefix, m2, "local", None)
                            add_edge(G, "", m1, func_prefix, m2, line, "data")
                    elif re.match(rgx.local_id, m1):
                        # val to store is a local id
                        add_node(G, func_prefix, m1, "local", None)
                        if re.match(rgx.global_id, m2):
                            if (
                                m2 in functions_declared_in_file
                                or m2[1:] in functions_defined_in_file.keys()
                            ):
                                add_node(G, "", m2, "global", None)
                                add_edge(G, "", glob_ref, "", m2, line, "path")
                            add_edge(G, func_prefix, m1, "", m2, line, "data")
                        else:
                            add_node(G, func_prefix, m2, "local", None)
                            add_edge(G, func_prefix, m1, func_prefix, m2, line, "data")
                    else:  # re.match(r'(' + rgx.immediate_value_or_undef + r'|null)', m1):
                        if re.match(rgx.global_id, m2):
                            if (
                                m2 in functions_declared_in_file
                                or m2[1:] in functions_defined_in_file.keys()
                            ):
                                add_node(G, "", m2, "global", None)
                                add_edge(G, "", glob_ref, "", a, line, "path")
                            add_edge(G, "", block_ref, "", m2, line, "data")
                        else:
                            add_node(G, func_prefix, m2, "local", None)
                            add_edge(G, "", block_ref, func_prefix, m2, line, "data")

            ############################################################################################################
            # (indirect) branch
            elif re.match("(indirect)?br ", line):

                # Unconditional branch
                if re.match("br label ", line):

                    # Get the label and add node
                    label_ = re.search(r"label (" + rgx.local_id + r")", line)
                    assert label_ is not None, "Could not identify label in:\n" + line
                    label = label_.group(1)
                    add_node(G, func_prefix, label, "label", ids_in_basic_block)

                    # Get the sink nodes of this basic block
                    # (sink nodes) --[stmt]--> (label)
                    added_edge = False
                    if len(ids_in_basic_block) > 0:
                        for n in list(set(ids_in_basic_block)):
                            if basic_block_leaf(G, n, ids_in_basic_block):
                                add_edge(G, "", n, func_prefix, label, line, "ctrl")
                                added_edge = True
                        if not added_edge:
                            add_edge(
                                G,
                                "",
                                list(set(ids_in_basic_block))[-1],
                                func_prefix,
                                label,
                                line,
                                "ctrl",
                            )

                    else:
                        # there are no local ids in this basic block, connect to the block reference
                        add_edge(G, "", block_ref, func_prefix, label, line, "ctrl")

                # Conditional branch
                elif re.match("br i1 ", line):

                    # Get all components
                    m = re.match(
                        r"br i1 (.*), label ("
                        + rgx.local_id
                        + r"), label ("
                        + rgx.local_id
                        + r")",
                        line,
                    )
                    assert m is not None, (
                        "Could not match components of statement:\n" + line
                    )
                    comparator = m.group(1)
                    labelT = m.group(2)
                    add_node(G, func_prefix, labelT, "label", ids_in_basic_block)
                    labelF = m.group(3)
                    add_node(G, func_prefix, labelF, "label", ids_in_basic_block)

                    # Check whether the comparator is a local identifier
                    if re.match(rgx.local_id, comparator):

                        if func_prefix + comparator not in ids_in_basic_block:
                            # If the statement has no parent in the present basic block, connect the paths
                            # (block ref) --[stmt]--> (assignee)
                            add_node(
                                G, func_prefix, comparator, "local", ids_in_basic_block
                            )
                            add_edge(
                                G, "", block_ref, func_prefix, comparator, line, "path"
                            )

                        # (comparator) --[stmt]--> (labels)
                        add_edge(
                            G,
                            func_prefix,
                            comparator,
                            func_prefix,
                            labelT,
                            line,
                            "ctrl",
                        )
                        add_edge(
                            G,
                            func_prefix,
                            comparator,
                            func_prefix,
                            labelF,
                            line,
                            "ctrl",
                        )

                    elif (
                        re.match(rgx.immediate_value, comparator)
                        or comparator == "undef"
                    ):
                        # Get the sink nodes of this basic block
                        # (sink nodes) --[stmt]--> (label)
                        added_edge = False
                        if len(ids_in_basic_block) > 0:
                            for n in ids_in_basic_block:
                                if basic_block_leaf(G, n, ids_in_basic_block):
                                    add_edge(
                                        G, "", n, func_prefix, labelT, line, "ctrl"
                                    )
                                    add_edge(
                                        G, "", n, func_prefix, labelF, line, "ctrl"
                                    )
                                    added_edge = True
                            assert added_edge, (
                                "No edge was added for statement:\n" + line
                            )
                        else:
                            # there are no local ids in this basic block, connect to the block reference
                            add_edge(
                                G, "", block_ref, func_prefix, labelT, line, "ctrl"
                            )
                            add_edge(
                                G, "", block_ref, func_prefix, labelF, line, "ctrl"
                            )

                    elif re.search(rgx.global_id, comparator):
                        # Get the comparator
                        m = re.search(rgx.global_id, comparator).group(0)

                        # If the statement has no parent in the present basic block, connect the paths
                        # (block ref) --[stmt]--> (assignee)
                        add_node(G, "", comparator, "local", None)
                        add_edge(G, "", block_ref, "", comparator, line, "path")

                        # (comparator) --[stmt]--> (labels)
                        add_edge(G, "", comparator, func_prefix, labelT, line, "ctrl")
                        add_edge(G, "", comparator, func_prefix, labelF, line, "ctrl")

                        # (block ref) --[stmt]--> (assignee)
                        add_edge(G, "", block_ref, "", comparator, line, "path")

                    else:

                        m = re.search(r"(" + rgx.global_id + r")", comparator)
                        if m is not None:
                            comparator = m.group(1)
                        else:
                            assert False, (
                                "Could not identify comparator in:\n"
                                + line
                                + "\nComparator:\n"
                                + comparator
                            )

                # Indirect branch
                elif re.match("indirectbr ", line):

                    # eg indirectbr i8* %18, [label %1639, label %754, ..., label %1303, label %1314]
                    # Get address
                    m = re.match(r"indirectbr .* (.*), \[", line)
                    assert m is not None, "Could not identify address in stmt:\n" + line
                    address = m.group(1)
                    assert re.match(rgx.local_id, address) is not None, (
                        "Address is not a local id in stmt:\n"
                        + line
                        + "\nAddress:\n"
                        + address
                    )
                    # Get labels
                    mlab = re.findall(r"label (.*)[,\]]", line)
                    mlab = mlab[0].replace("label", "")
                    mlab = mlab.split(",")
                    assert len(mlab) > 0, "Could not identify labels in stmt:\n" + line
                    labels = list()
                    for m in mlab:
                        labels.append(m.strip())

                    if func_prefix + address not in ids_in_basic_block:
                        # If the statement has no parent in the present basic block, connect the paths
                        # (block ref) --[stmt]--> (address)
                        add_edge(G, "", block_ref, func_prefix, address, line, "path")

                    for label in labels:
                        # (address) --[stmt]--> (labels)
                        add_node(G, func_prefix, label, "label", None)
                        add_edge(
                            G, func_prefix, address, func_prefix, label, line, "ctrl"
                        )

                else:
                    lines_not_added_to_graph.append(line)
                    assert False, "Could not recognize statement:\n" + line

            ############################################################################################################
            # switch
            elif re.match("switch ", line):

                # Get the comparator
                m = re.match(r"switch .* (.*), label (.*) \[", line)
                assert m is not None, (
                    "Could not match comparator of switch statement:\n" + line
                )
                comparator = m.group(1)
                deflabel = m.group(2)

                # Get the labels
                modif_line = line.replace("\n", " ")
                switchlist_ = re.search(r"\[.*\]$", modif_line)
                assert switchlist_ is not None, (
                    "Could not identify switch list in:\n" + line
                )
                switchlist = switchlist_.group(0)
                m_ = re.findall(
                    rgx.base_type
                    + " ("
                    + rgx.immediate_or_local_id
                    + r"), label ("
                    + rgx.local_id
                    + r")",
                    switchlist,
                )
                assert m_ is not None, (
                    "Could not match components of switch statement:\n" + line
                )
                vals = list()
                labels = list()
                for m in m_:
                    vals.append(m[0])
                    labels.append(m[1])

                # Get the default label and add it to the labels
                labels.append(deflabel)

                if re.match(rgx.local_id, comparator):
                    # Treat like an unconditional branch
                    sink_nodes = list()
                    if len(ids_in_basic_block) > 0:
                        for n in list(set(ids_in_basic_block)):
                            if basic_block_leaf(G, n, ids_in_basic_block):
                                sink_nodes.append(n)
                    for label in labels:
                        # (sink nodes) --[stmt]--> (label)
                        add_node(G, func_prefix, label, "label", None)
                        if len(sink_nodes) > 0:
                            for n in sink_nodes:
                                add_edge(G, "", n, func_prefix, label, line, "ctrl")
                        else:
                            # there are no local ids in this basic block, connect to the block reference
                            add_edge(G, "", block_ref, func_prefix, label, line, "ctrl")

                else:

                    for label in labels:
                        # (comparator) --[stmt]--> (labels)
                        add_node(G, func_prefix, label, "label", None)
                        add_edge(
                            G, func_prefix, comparator, func_prefix, label, line, "ctrl"
                        )

            ############################################################################################################
            # function call
            elif re.match(r"(" + rgx.local_id + r" = )?(tail )?(call|invoke) ", line):

                # Get function name
                if " asm " in line:
                    if (
                        line
                        == '%13 = tail call { %struct.rw_semaphore*, i64 } asm sideeffect "'
                    ):
                        line = '%13 = tail call { %struct.rw_semaphore*, i64 } asm sideeffect "# beginning down_read\0A\09.pushsection .smp_locks,\22a\22\0A.balign 4\0A.long 671f - .\0A.popsection\0A671:\0A\09lock;  incq ($3)\0A\09  jns        1f\0A  call call_rwsem_down_read_failed\0A1:\0A\09# ending down_read\0A\09", "=*m,={ax},={rsp},{ax},*m,2,~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"(%struct.atomic64_t* %11, %struct.rw_semaphore* %10, %struct.atomic64_t* %11, i64 %12) #4, !srcloc !9'
                    if line == '%16 = tail call i64 asm sideeffect "':
                        line = '%16 = tail call i64 asm sideeffect "# beginning __up_read\0A\09.pushsection .smp_locks,\22a\22\0A.balign 4\0A.long 671f - .\0A.popsection\0A671:\0A\09lock;   xadd      $1,($2)\0A\09  jns        1f\0A\09  call call_rwsem_wake\0A1:\0A# ending __up_read\0A", "=*m,={dx},{ax},1,*m,~{memory},~{cc},~{dirflag},~{fpsr},~{flags}"(%struct.atomic64_t* %11, %struct.rw_semaphore* %10, i64 -1, %struct.atomic64_t* %11) #4, !srcloc !11'
                    func_name_ = re.search(r" asm (?:sideeffect )?(\".*\")\(", line)
                else:
                    func_name_ = re.search(
                        r"(" + rgx.func_name + r")( to .*)?\(.*\)($|\n)", line
                    )
                if func_name_ is None:
                    func_name_ = re.search(r"(" + rgx.local_id + r")\(.*\)($|\n)", line)
                assert func_name_ is not None, (
                    "Could not identify function name in:\n" + line
                )
                func_name = func_name_.group(1)

                # If there is an assignee, get it and add its node
                if re.match(rgx.local_id + r" = ", line) is not None:
                    assignee = re.match(r"(" + rgx.local_id + r") = ", line).group(1)
                    add_node(G, func_prefix, assignee, "local", ids_in_basic_block)
                else:
                    assignee = None

                # Get string of arguments
                num_args, arg_list = get_num_args_func(line, func_name)

                # Get list of arguments
                if num_args > 0:

                    # Process argument string
                    s = re.search(r"(\([^\(\)]+\))", arg_list)
                    arg_list_modif = arg_list
                    arg_list_modif = re.sub(r"\{.*\}", "", arg_list_modif)
                    while s is not None:
                        if re.match(
                            r"\([^\(\)]+" + rgx.global_id + "[^\(\)]*\)", s.group(1)
                        ):
                            r = re.match(
                                r"\([^\(\)]+(" + rgx.global_id + ")[^\(\)]*\)",
                                s.group(1),
                            ).group(1)
                            arg_list_modif = arg_list_modif.replace(s.group(1), r, 1)
                        elif re.match(
                            r"\([^\(\)]+" + rgx.local_id + "[^\(\)]*\)", s.group(1)
                        ):
                            r = re.match(
                                r"\([^\(\)]+(" + rgx.local_id + ")[^\(\)]*\)",
                                s.group(1),
                            ).group(1)
                            arg_list_modif = arg_list_modif.replace(s.group(1), r, 1)
                        elif re.match(
                            r"\([^\(\)]+" + rgx.immediate_value + "[^\(\)]*\)",
                            s.group(1),
                        ):
                            arg_list_modif = arg_list_modif.replace(
                                s.group(1), str(1000), 1
                            )
                        else:
                            arg_list_modif = arg_list_modif.replace(s.group(1), "", 1)
                        # For next loop
                        s = re.search(r"(\([^\(\)]+\))", arg_list_modif)

                    # Get "to match"
                    to_match = arg_list_modif.split(",")
                    while len(to_match) != num_args:
                        to_modif = re.search("\((.*),(.*)\)", arg_list)
                        if to_modif is None:
                            to_modif = re.search('"(.*),(.*)"', arg_list)
                        assert to_modif is not None, (
                            "Unexpected number of arguments in " + line
                        )
                        for i in range(1, int(len(to_modif.groups())), 2):
                            arg_list = arg_list.replace(
                                to_modif.group(i) + "," + to_modif.group(i + 1),
                                to_modif.group(i) + " " + to_modif.group(i + 1),
                            )
                        to_match = arg_list.split(",")
                    assert len(to_match) == num_args, (
                        "Wrong number of arguments in line "
                        + line
                        + " (args: "
                        + arg_list
                        + ", expected "
                        + str(num_args)
                        + " arguments."
                    )

                    # Get all individual arguments (can contain immediates!)
                    # (these are the called arguments, not the defined ones!)
                    args = list()
                    try:
                        for t in to_match:

                            if (
                                re.search(rgx.local_id + r"(?!\* )(?=([\s,\)]|$))", t)
                                is not None
                            ):
                                args.append(
                                    re.search(
                                        rgx.local_id + r"(?!\)\* )(?=([\s,]|$))", t
                                    ).group(0)
                                )
                            elif re.search(rgx.global_id, t) is not None:
                                args.append(re.search(rgx.global_id, t).group(0))
                            elif (
                                re.search(rgx.immediate_value_or_undef + r"$", t)
                                is not None
                            ):
                                args.append(
                                    re.search(rgx.immediate_value_or_undef, t).group(0)
                                )
                            elif (
                                re.search(
                                    rgx.immediate_or_local_id_or_undef
                                    + r"(?!\* )(?=\()",
                                    t,
                                )
                                is not None
                            ):
                                # eg call void @llvm.dbg.value(metadata i32 %call())
                                # eg call void @llvm.dbg.value(metadata i32 0())
                                args.append(
                                    re.search(
                                        rgx.immediate_or_local_id_or_undef
                                        + r"(?!\)\* )(?=\()",
                                        t,
                                    ).group(0)
                                )
                            elif re.search(r"\s+$", t) is not None or "metadata" in t:
                                args.append(str(1000))
                            else:
                                print(
                                    "Could not properly identify argument in: \n"
                                    + line
                                    + "\n"
                                    "argument: "
                                    + str(t)
                                    + ",\n"
                                    + "argument list: "
                                    + str(arg_list)
                                    + "\n"
                                    + "argument list (modif): "
                                    + str(arg_list_modif)
                                )
                                raise ValueError("FunctionNotSupported")
                    except ValueError:
                        raise

                if func_name[1:] in functions_defined_in_file.keys():
                    # if this function is defined in this file

                    # Get the called function's shortened name
                    func_key = func_name[1:]
                    called_func = functions_defined_in_file[func_key][0]

                    # Get the list of defined (*NOT* called!) arguments
                    if len(functions_defined_in_file[func_key]) == 5:
                        # Arguments are named explicitely
                        args_defined = functions_defined_in_file[func_key][4]
                    else:
                        # Arguments are named implicitely
                        args_defined = ["%" + str(i) for i in range(0, num_args)]

                    # Connect called->defined arguments
                    if num_args > 0:
                        # (arg called) --[stmt]--> (arg defined)
                        no_parent = True
                        for a_called, a_defined in zip(args, args_defined):
                            add_node(G, called_func + "_", a_defined, "local", None)
                            if re.match(rgx.global_id, a_called):
                                if (
                                    a_called in functions_declared_in_file
                                    or a_called[1:] in functions_defined_in_file.keys()
                                ):
                                    add_node(G, "", a_called, "global", None)
                                    add_edge(
                                        G, "", glob_ref, "", a_called, line, "path"
                                    )
                                add_edge(
                                    G,
                                    "",
                                    a_called,
                                    called_func + "_",
                                    a_defined,
                                    line,
                                    "data",
                                )
                            elif re.match(rgx.local_id, a_called):
                                # if operand is in this basic block, then the statement has a parent
                                if func_prefix + a_called in ids_in_basic_block:
                                    no_parent = False
                                add_node(G, func_prefix, a_called, "local", None)
                                add_edge(
                                    G,
                                    func_prefix,
                                    a_called,
                                    called_func + "_",
                                    a_defined,
                                    line,
                                    "data",
                                )

                    else:
                        # There is no data flow, so we introduce control flow
                        # (block ref) --[stmt]--> (defined function's first block ref)
                        add_node(G, called_func + "_", "#top_ref", "label", None)
                        add_edge(
                            G,
                            "",
                            block_ref,
                            called_func + "_",
                            "#top_ref",
                            line,
                            "ctrl",
                        )
                        no_parent = False

                    # If there is an assignee
                    if assignee is not None:

                        # Get return statement
                        ret_ = functions_defined_in_file[func_key][2]
                        ret_node_match = re.match(
                            r"ret .* (%?" + rgx.local_id_no_perc + r"|false|true)", ret_
                        )
                        assert ret_node_match is not None, (
                            "Return statement could not be identified in "
                            + "function:\n"
                            + line
                            + "\nreturn stmt:\n"
                            + ret_
                        )
                        ret_node = ret_node_match.group(1)

                        # Connect
                        # (returned value) --[stmt]--> (assignee)
                        if re.match(rgx.local_id, ret_node):
                            ret = ret_node
                        else:
                            assert ret_node != "void", (
                                "Void return node found in call with assignment:\n"
                                + line
                            )
                            ret = "#ret"
                        add_node(G, called_func + "_", ret, "local", None)
                        add_edge(
                            G,
                            called_func + "_",
                            ret,
                            func_prefix,
                            assignee,
                            line,
                            "data",
                        )

                        # If the statement has no parent in the present basic block, connect the paths
                        if no_parent:
                            # (block ref) --[stmt]--> (assignee)
                            add_edge(
                                G, "", block_ref, func_prefix, assignee, line, "path"
                            )

                    else:  # no assignee

                        # If the statement has no parent in the present basic block, connect the paths
                        if no_parent:
                            # (block ref) --[stmt]--> (ad hoc)
                            ad_hoc_count = add_edge_dummy(
                                G, "", block_ref, line, ad_hoc_count
                            )

                    # if it is an "invoke"
                    if re.match(rgx.local_id + r" = (tail )?( fastcc)?invoke", line):
                        # Connect "void" to the normal and unwind labels in invocation
                        m_loc, m_glob, m_label, m_label2 = get_identifiers_from_line(
                            line
                        )
                        assert len(m_label) == 2, (
                            "Could not identify 2 labels in:\n" + line
                        )
                        add_node(
                            G, func_prefix, m_label[0], "label", ids_in_basic_block
                        )
                        add_node(
                            G, func_prefix, m_label[1], "label", ids_in_basic_block
                        )
                        add_edge(
                            G,
                            called_func + "_",
                            ret,
                            func_prefix,
                            m_label[0],
                            line,
                            "ctrl",
                        )
                        add_edge(
                            G,
                            called_func + "_",
                            ret,
                            func_prefix,
                            m_label[1],
                            line,
                            "ctrl",
                        )
                    elif re.match(
                        r"invoke (.* )?void", line
                    ):  # if it is an "invoke void"
                        # Connect "void" to the normal and unwind labels in invocation
                        ret = "#ret"
                        add_node(G, called_func + "_", ret, "ret", None)
                        m_loc, m_glob, m_label, m_label2 = get_identifiers_from_line(
                            line
                        )
                        assert len(m_label) == 2, (
                            "Could not identify 2 labels in:\n" + line
                        )
                        add_node(
                            G, func_prefix, m_label[0], "label", ids_in_basic_block
                        )
                        add_node(
                            G, func_prefix, m_label[1], "label", ids_in_basic_block
                        )
                        add_edge(
                            G,
                            called_func + "_",
                            ret,
                            func_prefix,
                            m_label[0],
                            line,
                            "ctrl",
                        )
                        add_edge(
                            G,
                            called_func + "_",
                            ret,
                            func_prefix,
                            m_label[1],
                            line,
                            "ctrl",
                        )

                else:
                    # if this function is *NOT* defined in this file (i.e. we don't know its body)
                    # - only declared in this file
                    # - call .* asm
                    # - function pointer
                    # - intrinsic

                    #  Connect arguments
                    no_parent = True
                    a_connected = ""
                    if assignee is None:
                        no_parent = True
                        if num_args > 0:
                            for a in args:
                                if not re.match(rgx.immediate_value_or_undef, a):
                                    if (
                                        a[0] != "@"
                                        and func_prefix + a in ids_in_basic_block
                                    ):
                                        no_parent = False
                                    # (arg) --[stmt]--> (dummy_node)
                                    a_connected = a
                                    if re.match(rgx.global_id, a):
                                        if (
                                            a in functions_declared_in_file
                                            or a[1:] in functions_defined_in_file.keys()
                                        ):
                                            add_node(G, "", a, "global", None)
                                            add_edge(
                                                G, "", glob_ref, "", a, line, "path"
                                            )
                                        ad_hoc_count = add_edge_dummy(
                                            G, "", a, line, ad_hoc_count
                                        )
                                    else:
                                        add_node(G, func_prefix, a, "local", None)
                                        ad_hoc_count = add_edge_dummy(
                                            G, func_prefix, a, line, ad_hoc_count
                                        )
                        if no_parent:
                            # (block ref) --[stmt]--> (assignee)
                            if (
                                a_connected != ""
                                and a_connected != "null"
                                and re.match(rgx.immediate_value_or_undef, a_connected)
                                is None
                            ):
                                if re.match(rgx.global_id, a_connected):
                                    add_edge(
                                        G, "", block_ref, "", a_connected, line, "path"
                                    )
                                else:
                                    add_edge(
                                        G,
                                        "",
                                        block_ref,
                                        func_prefix,
                                        a_connected,
                                        line,
                                        "path",
                                    )
                            else:
                                ad_hoc_count = add_edge_dummy(
                                    G, "", block_ref, line, ad_hoc_count
                                )

                        if re.match(r"invoke void", line):  # if it is an "invoke void"
                            # Connect "void" to the normal and unwind labels in invocation
                            (
                                m_loc,
                                m_glob,
                                m_label,
                                m_label2,
                            ) = get_identifiers_from_line(line)
                            assert len(m_label) == 2, (
                                "Could not identify 2 labels in:\n" + line
                            )
                            add_node(
                                G, func_prefix, m_label[0], "label", ids_in_basic_block
                            )
                            add_node(
                                G, func_prefix, m_label[1], "label", ids_in_basic_block
                            )
                            add_edge(
                                G, "", block_ref, func_prefix, m_label[0], line, "ctrl"
                            )
                            add_edge(
                                G, "", block_ref, func_prefix, m_label[1], line, "ctrl"
                            )

                    else:
                        no_parent = True
                        if num_args > 0:
                            for a in args:
                                if not re.match(rgx.immediate_value_or_undef, a):
                                    if (
                                        a[0] != "@"
                                        and func_prefix + a in ids_in_basic_block
                                    ):
                                        no_parent = False
                                    # (arg) --[stmt]--> (assignee)
                                    if re.match(rgx.global_id, a):
                                        if (
                                            a in functions_declared_in_file
                                            or a[1:] in functions_defined_in_file.keys()
                                        ):
                                            add_node(G, "", a, "global", None)
                                            add_edge(
                                                G, "", glob_ref, "", a, line, "path"
                                            )
                                        add_edge(
                                            G,
                                            "",
                                            a,
                                            func_prefix,
                                            assignee,
                                            line,
                                            "data",
                                        )
                                    else:
                                        add_node(G, func_prefix, a, "local", None)
                                        add_edge(
                                            G,
                                            func_prefix,
                                            a,
                                            func_prefix,
                                            assignee,
                                            line,
                                            "data",
                                        )
                        # If the statement has no parent in the present basic block, connect the paths
                        if no_parent:
                            # (block ref) --[stmt]--> (assignee)
                            add_edge(
                                G, "", block_ref, func_prefix, assignee, line, "path"
                            )

                        # if it is an "invoke"
                        if re.match(
                            rgx.local_id + r" = (tail )?( fastcc)?invoke", line
                        ):
                            (
                                m_loc,
                                m_glob,
                                m_label,
                                m_label2,
                            ) = get_identifiers_from_line(line)
                            assert len(m_label) == 2, (
                                "Could not identify 2 labels in:\n" + line
                            )
                            add_node(
                                G, func_prefix, m_label[0], "label", ids_in_basic_block
                            )
                            add_node(
                                G, func_prefix, m_label[1], "label", ids_in_basic_block
                            )
                            add_edge(
                                G,
                                func_prefix,
                                assignee,
                                func_prefix,
                                m_label[0],
                                line,
                                "ctrl",
                            )
                            add_edge(
                                G,
                                func_prefix,
                                assignee,
                                func_prefix,
                                m_label[1],
                                line,
                                "ctrl",
                            )

            ############################################################################################################
            # return statement
            elif re.match("ret ", line):

                # Identify the returned value and get a node for it if needed
                if re.match(
                    r"ret .* " + rgx.immediate_value_or_undef, line
                ) or re.match(r"ret void", line):

                    # If it is an immediate of some sorts
                    add_node(G, func_prefix, "#ret", "imm", None)
                    added_edge = False
                    if len(ids_in_basic_block) > 0:
                        for n in ids_in_basic_block:
                            if basic_block_leaf(G, n, ids_in_basic_block):
                                # (leaf in basic block) --[stmt]--> (ad hoc)
                                add_edge(G, "", n, func_prefix, "#ret", line, None)
                                added_edge = True
                        if not added_edge:
                            add_edge(G, "", block_ref, func_prefix, "#ret", line, None)
                    else:
                        # there are no local ids in this basic block, connect to the block reference
                        add_edge(G, "", block_ref, func_prefix, "#ret", line, None)

                elif re.match(
                    r"ret .* " + rgx.local_id + r"(?!\* )(?=([\s,\)]|$))", line
                ):

                    # Match the identifier
                    m = re.match(
                        r"ret .* (" + rgx.local_id + r")(?!\* )(?=([\s,\)]|$))", line
                    )
                    ret = m.group(1)
                    # (id) --[stmt]--> (ad hoc)
                    ad_hoc_count = add_edge_dummy(
                        G, func_prefix, ret, line, ad_hoc_count
                    )

                    # If the statement has no parent in the present basic block, connect the paths
                    if func_prefix + ret not in ids_in_basic_block:
                        # (block ref) --[stmt]--> (assignee)
                        add_edge(G, "", block_ref, func_prefix, ret, line, "path")

                elif re.match(r"ret .* " + rgx.global_id, line):

                    # Match the identifier
                    m = re.match(r"ret .* (" + rgx.global_id + r")", line)
                    ret = m.group(1)
                    # (id) --[stmt]--> (ad hoc)
                    if (
                        ret in functions_declared_in_file
                        or ret[1:] in functions_defined_in_file.keys()
                    ):
                        add_node(G, "", ret, "global", None)
                        add_edge(G, "", glob_ref, "", ret, line, "path")
                    ad_hoc_count = add_edge_dummy(G, "", ret, line, ad_hoc_count)

                    # If the statement has no parent in the present basic block, connect the paths
                    if func_prefix + ret not in ids_in_basic_block:
                        # (block ref) --[stmt]--> (assignee)
                        add_edge(G, "", block_ref, "", ret, line, "path")

                else:

                    assert False, "Could not identify returned value in:\n" + line

            ############################################################################################################
            # return statement
            elif re.match("resume ", line):

                # Get the identifier
                m = re.match(r"resume .* (.*)$", line)
                assert m is not None, "Could not identify the variable in:\n" + line
                var = m.group(1)
                assert re.match(rgx.local_id, var) is not None, (
                    "Variable in resume is not a local id:\n"
                    + line
                    + "\nVariable:\n"
                    + var
                )

                # Add edge
                ad_hoc_count = add_edge_dummy(G, func_prefix, var, line, ad_hoc_count)

                # If the statement has no parent in the present basic block, connect the paths
                if func_prefix + var not in ids_in_basic_block:
                    # (block ref) --[stmt]--> (var)
                    add_edge(G, "", block_ref, func_prefix, var, line, "path")

            ############################################################################################################
            # unreachable or fence
            elif re.search("(unreachable|fence)", line):

                # Add edge
                ad_hoc_count = add_edge_dummy(G, "", block_ref, line, ad_hoc_count)

            ############################################################################################################
            # Default
            else:
                lines_not_added_to_graph.append(line)
                assert False, (
                    "Could not recognize statement on line " + str(i) + ":\n" + line
                )

    # Put things together
    for k, v in functions_defined_in_file.items():
        if v[1]:
            # If this function is called
            # Connect to the corresponding #top_refs if there are any
            func_prefix = v[0] + "_"
            if (func_prefix + "#top_ref") in list(G.nodes()):
                G = nx.contracted_nodes(
                    G, func_block_refs[func_prefix], func_prefix + "#top_ref", False
                )

    return G


def check_vocabulary_size(preprocessed_file, G):
    """
    Make sure the vocabulary size in the graph representation matches the one in the text representation
    :param preprocessed_file: list of statements contained in the preprocessed file
    :param G: Graph
    """
    # Construct text-vocabulary
    vocabulary_after_preprocessing = sorted(set(preprocessed_file))
    for i, t in enumerate(vocabulary_after_preprocessing):  # Remove 'label' statements
        if re.match(rgx.start_basic_block, t):
            vocabulary_after_preprocessing[i] = "REMOVE"
    vocabulary_after_preprocessing = [
        t for t in vocabulary_after_preprocessing if t != "REMOVE"
    ]
    vocabulary_size_after_preprocessing = len(vocabulary_after_preprocessing)

    # Construct graph-vocabulary
    vocabulary_after_graph_construction = sorted(
        set([e[2]["stmt"] for e in G.edges(data=True)])
    )
    vocabulary_size_after_graph_construction = len(vocabulary_after_graph_construction)

    # Perform checks
    try:
        if (
            vocabulary_size_after_graph_construction
            != vocabulary_size_after_preprocessing
        ):
            source_data_after_preprocessing_file = "vocabulary_text.txt"
            source_data_after_graph_construction_file = "vocabulary_graph.txt"
            source_data_after_graph_construction = list(
                vocabulary_after_graph_construction
            )
            print_data(
                vocabulary_after_preprocessing, source_data_after_preprocessing_file
            )
            print_data(
                source_data_after_graph_construction,
                source_data_after_graph_construction_file,
            )
            print(
                "There are ",
                vocabulary_size_after_preprocessing
                - vocabulary_size_after_graph_construction,
                " words more in the text representation than in the graph representation",
            )
            in_graph_not_text = set(vocabulary_after_graph_construction) - set(
                vocabulary_after_preprocessing
            )
            if in_graph_not_text:
                print(
                    "The following words are in the graph representation but not in the text representation: "
                )
                for s in in_graph_not_text:
                    print("\t", s)
                print(
                    "The words above are in the graph representation but not in the text representation: "
                )
                raise ValueError("GraphMisconstructed")
            in_text_not_graph = list(
                set(vocabulary_after_preprocessing)
                - set(vocabulary_after_graph_construction)
            )
            if in_text_not_graph:
                print(
                    "The following words are in the text representation but not in the graph representation: "
                )
                for s in in_text_not_graph:
                    print("\t", s)
                print(
                    "The words above are in the text representation but not in the graph representation: "
                )
                raise ValueError("GraphMisconstructed")
    except ValueError:
        return None


def CheckGraphOrDie(G, filename):
    """Assess the validity of the construction of this graph using following
    criteria:
      - Does every node have an ID?
      - Have all statements been added to the graph?
      - Are there any isolated nodes?
      - Are there any global nodes which are leaf nodes (i.e. are not re-used)
      - Make a list of multi-edges
      - Make sure the graph is connected
    :param G: context graph to be checked
    :param filename: name of file
    :return: multi-edges: list of edges for which parallel edges exist
             G
    """
    # Make sure each node has an id
    for n in sorted(list(G.nodes(data=True))):
        assert n[1], 'Node "' + n[0] + '" has no id (file ' + filename + ")"

    # Make sure there are no isolated nodes
    isolated_nodes = [n for n in G.nodes() if all_degrees(G, n) == 0]
    isolated_nodes = sorted(isolated_nodes, key=sort_key)
    if len(isolated_nodes) > 0:
        print("\nIsolated nodes: ")
        for n in isolated_nodes:
            print(n)
        assert False, (
            "Isolated nodes found in construction of context graph for file " + filename
        )

    # Make sure there is only one root node
    root_nodes = [n for n in G.nodes() if G.out_degree(n) > 0 and G.in_degree(n) == 0]
    if len(root_nodes) != 1:
        # Found more than one root node
        for n in root_nodes:
            print(n)
            print("\twith edges", G.edges(n, data=True))
        # assert False, "Found more than one root node"
        print("Found more than one root node")

    # Make sure there are no leaf nodes which are labels
    leaf_nodes = [
        n
        for n in G.nodes(data=True)
        if G.out_degree(n[0]) == 0 and G.in_degree(n[0]) >= 1
    ]
    if len(leaf_nodes) > 0:
        label_leaf = False
        for l in leaf_nodes:
            if l[1]["id"] == "label":
                print(l)
                label_leaf = True
        if label_leaf:
            # assert False, "Found label nodes that are leaves"
            print("Found label nodes that are leaves")

    # Make a list of all multi-edges (i.e. edges for which there exists another edge which connects the same nodes)
    multi_edges = dict()
    for e in G.edges(data=True):
        if (
            G.number_of_edges(e[0], e[1]) > 1
            and (e[0] + " ||| " + e[1]) not in multi_edges.keys()
        ):
            edges_ = G[e[0]][e[1]]
            s1 = edges_[0]["stmt"]
            s_ = list()
            s_.append(s1)
            for i in range(1, len(edges_)):
                if edges_[i]["stmt"] not in s_:
                    s_.append(edges_[i]["stmt"])
            if len(s_) > 1:
                multi_edges[e[0] + " ||| " + e[1]] = s_

    # Make sure the graph is not disconnected
    G_undirected = G.to_undirected()
    try:
        if not nx.is_connected(G_undirected):
            # Get the smallest connected component
            print(
                "\nNumber of connected components: ",
                nx.number_connected_components(G_undirected),
                "\n",
            )
            cc_ = sorted(nx.connected_components(G_undirected), key=len)
            print("\nSecondary (non-main) connected components: \n")
            for cc in cc_[:-1]:
                print("\n\t--- Connected component with ", len(cc), " nodes \n")
                for e in G_undirected.edges(cc, data=True):
                    print("\tnode 1: ", e[0])
                    print("\tnode 2: ", e[1])
                    print(e[2]["stmt"])
                    if e[0] in list(G.nodes):
                        G.remove_node(e[0])
                    if e[1] in list(G.nodes):
                        G.remove_node(e[1])
            print("WARNING! Graph for file " + filename + " is disconnected")
            raise ValueError("GraphMisconstructed")
    except ValueError:
        raise

    # Return the list of multi-edges
    return multi_edges, G


def BuildContextualFlowGraph(llvm_lines, functions_declared_in_file, filename):
    """
    Given a file of source code, construct a context graph
    This function is called once for each file

    :param llvm_lines: LLVM-IR source file (list of strings containing lines of LLVM
        IR).
    :param functions_declared_in_file:
    :param filename: name of file
    :return: A <digraph, multi_edge_list> tuple, where <digraph> is a directed
      graph in which nodes are identifiers or ad-hoc and edges are statements
      which is meant as a representation of both data and flow control of the
      code capturing the notion of context; and <multi_edge_list> is a list of
      edges that have parallel edges.
    """
    # Create a graph
    graph = nx.MultiDiGraph()

    # Dictionary of functions defined in the file
    # keys: names of functions which are defined (not just declared) in this file
    # values: pair: [shortened function name, its corresponding return statement]
    functions_defined_in_file = construct_function_dictionary(llvm_lines)

    # Add lines to graph
    graph = add_stmts_to_graph(
        graph, llvm_lines, functions_defined_in_file, functions_declared_in_file
    )

    # Make sure the vocabulary size in the graph representation matches the one in the text representation
    check_vocabulary_size(llvm_lines, graph)

    # Make sure the graph was correctly constructed.
    multi_edges, graph = CheckGraphOrDie(graph, filename)

    return graph, multi_edges


def get_data_characteristics(data_folders):
    """
    Get data characteristics
    :param data_folders: data folders
    :return: boolean
    """
    # Determine whether the data set uses structure names with a pattern ('%"[^"]*") different from local identifiers
    specific_struct_name_pattern = False
    for folder in data_folders:
        if folder in [
            "data/testing/ll_1",
            "data/testing/ll_2",
            "data/testing/ll_20",
            "data/eigen/ll_350a",
            "data/eigen/ll_350b",
            "data/eigen/ll_1000",
            "data/eigen/eigen_addsub",
            "data/eigen/eigen_matdec",
            "data/eigen/eigen_matmul",
            "data/eigen/eigen_sparse",
            "data/eigen/eigen_vecops",
        ]:
            specific_struct_name_pattern = True

    # Return
    return specific_struct_name_pattern


def disambiguate_stmts(G):
    """
    Append a #number to statements in order to disambiguate them
    :param G: graph
    :return: disambiguated graph
    """
    # Helper dictionary
    appear_dic = dict()
    for ed in list(set([e[2] for e in all_edges(G, data=True)])):
        appear_dic[ed] = 0

    G_diff = copy.deepcopy(G)
    # Loop over the original graph's edges
    for e in G_diff.edges(data=True):

        # If it's a multi-edge, get the right index
        if G_diff.number_of_edges(e[0], e[1]) == 1:
            i = 0
        else:
            for i in range(G_diff.number_of_edges(e[0], e[1])):
                if G_diff[e[0]][e[1]][i]["stmt"] == e[2]["stmt"]:
                    break

        # Add target statement
        stmt_old = G_diff[e[0]][e[1]][i]["stmt"]
        stmt_new = stmt_old + "§" + str(appear_dic[stmt_old])
        G_diff[e[0]][e[1]][i]["stmt"] = stmt_new
        appear_dic[stmt_old] += 1

    return G_diff


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


def inline_struct_types_in_file(G, dic, specific_struct_name_pattern):
    """
    Inline structure types in the whole file
    :param G: graph of statements
    :param dic: dictionary ["structure name", "corresponding literal structure"]
    :param specific_struct_name_pattern: booleas
    :return: modified graph
    """
    to_track = ""
    # Inline the named structures throughout the file
    for e in G.edges(data=True):

        # For debugging
        if len(to_track) > 0:
            if e[2]["stmt"] == to_track:
                print("Found statement " + e[2]["stmt"])

        # As long as the stmt contains named structures/classes
        if specific_struct_name_pattern:
            m = re.search(r"(" + rgx.struct_name + r")", e[2]["stmt"])
            while m:
                # Replace them by their value in dictionary
                e[2]["stmt"] = re.sub(
                    r"(" + rgx.struct_name + r")", dic[m.group(1)], e[2]["stmt"]
                )
                m = re.search(r"(" + rgx.struct_name + r")", e[2]["stmt"])
        else:
            possible_struct = re.findall("(" + rgx.struct_name + ")", e[2]["stmt"])
            if len(possible_struct) > 0:
                for s in possible_struct:
                    if s in dic and not re.match(s + r"\d* = ", e[2]["stmt"]):
                        # Replace them by their value in dictionary
                        e[2]["stmt"] = re.sub(
                            re.escape(s) + rgx.struct_lookahead, dic[s], e[2]["stmt"]
                        )

    return G


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
        raise ValueError(e)


def inline_struct_types(
    G, data_with_struct_def, file_name, specific_struct_name_pattern
):
    """
    :param G: graph of statements
    :param data_with_struct_def: list of statements containing the structure definitions
    :param file_name: file name
    :param specific_struct_name_pattern: booleas
    :return: modified graph
    """
    # Print
    print("Inlining structures for file : ", file_name)

    # Construct a dictionary ["structure name", "corresponding literal structure"]
    data_with_struct_def, dict_temp = construct_struct_types_dictionary_for_file(
        data_with_struct_def
    )

    # If the dictionary is empty
    if not dict_temp:
        found_type = False
        for l in data_with_struct_def:
            if re.match(rgx.struct_name + " = type (<?\{ .* \}|opaque|{})", l):
                found_type = True
                break
        assert not found_type, (
            "Structures' dictionary is empty for file containing type definitions: \n"
            + data_with_struct_def[0]
            + "\n"
            + data_with_struct_def[1]
            + "\n"
            + data_with_struct_def
            + "\n"
        )

    # If the dictionary is not empty
    else:
        # Use the constructed dictionary to substitute named structures
        # by their corresponding literal structure throughout the program
        G = inline_struct_types_in_file(G, dict_temp, specific_struct_name_pattern)

    return G, dict_temp


def abstract_statements_from_identifiers(G):
    """
    Simplify lines of code by stripping them from their identifiers,
    unnamed values, etc. so that LLVM IR statements can be abstracted from them
    :param G: graph of statements
    :return: modified input data
    """
    for _, _, data in G.edges(data=True):
        data["stmt"] = PreprocessStatement(data["stmt"])
    return G


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


########################################################################################################################
# Dual-XFG-building
########################################################################################################################
def build_dual_graph(G):
    """
    :param G: base XFG-graph
    :return: dual of XFG graph
    """
    # Build dual graph
    D = nx.Graph()

    # Loop over the original graph's edges
    for e in all_edges(G, data=True):

        # Add target statement
        target_stmt = e[2]
        D.add_node(target_stmt)

        # Get its neighbour-statements
        neighbor_stmts = [f[2] for f in all_edges(G, [e[0], e[1]], data=True)]
        neighbor_stmts = list(set(neighbor_stmts))  # remove duplicates
        neighbor_stmts.remove(target_stmt)

        # Add to dual graph
        for s in neighbor_stmts:
            if D.has_edge(target_stmt, s):
                # this edge already exists, increment its weight
                D[target_stmt][s]["weight"] += 1
            else:
                # this edge does not yet exist, add it
                D.add_edge(target_stmt, s, weight=1)

    return D


def check_sanity(D, G):
    """
    Check construction of dual-XFG
    :param D: dual XFG
    :param G: base graph
    """
    isolated_nodes = [n for n in D.nodes() if D.degree(n) == 0]
    if len(isolated_nodes) != 0:
        print("WARNING! Isolated nodes found in D-graph")
        for n in isolated_nodes:
            D.remove_node(n)
        assert "Isolated nodes found in D-graph"
    if len(list(D.nodes)) != 0:
        if not nx.is_connected(D):
            print("WARNING! D-graph is disconnected")
            assert "D-graph is disconnected"
        if D.number_of_nodes() != G.number_of_edges():
            print(
                "WARNING! The number of nodes in the D-graph ("
                + str(D.number_of_nodes())
                + ") does not match the vnumber of edges in the G-graph ("
                + str(G.number_of_edges())
                + ")"
            )
            assert "Mismatch"


def CreateContextualFlowGraphsFromBytecodes(data_folder):
    """Construct XFGs (conteXtual Flow Graphs) from LLVM IR code.

    Input files:
        data_folder/*/*.ll

    Files produced:
        data_folder/*/data_read_pickle
        data_folder/*_preprocessed/data_preprocessed_pickle

    Folders produced:
        data_folder/*_preprocessed/data_transformed/
        data_folder/*_preprocessed/preprocessed/
        data_folder/*_preprocessed/structure_dictionaries/
        data_folder/*_preprocessed/xfg/
        data_folder/*_preprocessed/xfg_dual/

    :param data_folder: the path to the parent directory of the subfolders containing
        raw LLVM IR code.
    :return: List of subfolders containing raw LLVM IR code.
    """
    # Get raw data sub-folders
    assert os.path.exists(data_folder), "Folder " + data_folder + " does not exist"
    folders_raw = list()
    listing_to_explore = [
        os.path.join(data_folder, f) for f in os.listdir(data_folder + "/")
    ]
    while len(listing_to_explore) > 0:
        f = listing_to_explore.pop()
        if os.path.isdir(f):
            f_contents = os.listdir(f + "/")
            for file in f_contents:
                # if it contains raw .ll files
                if file[-3:] == ".ll":
                    folders_raw.append(f)
                    break
                elif os.path.isdir(os.path.join(f, file)):
                    listing_to_explore.append(os.path.join(f, file))

    print(
        "In folder",
        data_folder,
        ", found",
        len(folders_raw),
        "raw data folder(s):\n",
        folders_raw,
    )

    # Loop over raw data folders
    num_folders = len(folders_raw)
    for folder_counter, folder_raw in enumerate(folders_raw):

        # Print
        print(
            "\n------ Processing raw folder",
            folder_raw,
            "(",
            folder_counter + 1,
            "/",
            num_folders,
            ")",
        )

        ################################################################################################################
        # Check whether the folder has been preprocessed already R

        folder_preprocessed = folder_raw + "_preprocessed"
        data_preprocessing_done_filename = os.path.join(
            folder_preprocessed, "preprocessing.done"
        )
        if os.path.exists(data_preprocessing_done_filename):
            print(
                "\tfolder already preprocessed (found file",
                data_preprocessing_done_filename,
                ")",
            )

        else:  # this folder has not been preprocessed yet:

            ############################################################################################################
            # Read data from files

            data_read_from_folder_filename = os.path.join(
                folder_raw, "data_read_pickle"
            )
            if os.path.exists(data_read_from_folder_filename):
                # Load pre-processed data
                print(
                    "\n--- Loading data read from folder ",
                    folder_raw,
                    " from file ",
                    data_read_from_folder_filename,
                )
                with open(data_read_from_folder_filename, "rb") as f:
                    raw_data, file_names = pickle.load(f)

            else:

                # Read data from folder and pickle it
                print("\n--- Read data from folder ", folder_raw)
                raw_data, file_names = read_data_files_from_folder(folder_raw)
                print(
                    "Dumping data read from folder ",
                    folder_raw,
                    " into file ",
                    data_read_from_folder_filename,
                )
                i2v_utils.safe_pickle(
                    [raw_data, file_names], data_read_from_folder_filename
                )

            # Print data statistics and release memory
            source_data_list, source_data = data_statistics(
                raw_data, descr="reading data from source files"
            )
            del source_data_list

            ############################################################################################################
            # Pre-process source code

            if not os.path.exists(folder_preprocessed):
                os.makedirs(folder_preprocessed)
            data_preprocessed_filename = os.path.join(
                folder_preprocessed, "data_preprocessed_pickle"
            )
            if not os.path.exists(data_preprocessed_filename):

                # Source code transformation: simple pre-processing
                print("\n--- Pre-process code")
                preprocessed_data, functions_declared_in_files = preprocess(raw_data)
                preprocessed_data_with_structure_def = raw_data
                # save the vocabulary for later checks
                vocabulary_after_preprocessing = list(
                    set(collapse_into_one_list(preprocessed_data))
                )

                # Dump pre-processed data into a folder to be reused
                print(
                    "Writing pre-processed data into folder ", folder_preprocessed, "/"
                )
                print_preprocessed_data(
                    preprocessed_data, folder_preprocessed, file_names
                )
                print(
                    "Dumping pre-processed data info file ", data_preprocessed_filename
                )
                i2v_utils.safe_pickle(
                    [
                        preprocessed_data,
                        functions_declared_in_files,
                        preprocessed_data_with_structure_def,
                        vocabulary_after_preprocessing,
                    ],
                    data_preprocessed_filename,
                )

            else:

                # Load pre-processed data
                print(
                    "\n--- Loading pre-processed data from ", data_preprocessed_filename
                )
                with open(data_preprocessed_filename, "rb") as f:
                    (
                        preprocessed_data,
                        functions_declared_in_files,
                        preprocessed_data_with_structure_def,
                        vocabulary_after_preprocessing,
                    ) = pickle.load(f)

            # Print statistics and release memory
            source_data_list, source_data = data_statistics(
                preprocessed_data, descr="pre-processing code"
            )
            del source_data_list

            # Make sure folders exist
            graph_folder = os.path.join(folder_preprocessed, "xfg")
            if not os.path.exists(graph_folder):
                os.makedirs(graph_folder)
            structures_folder = os.path.join(
                folder_preprocessed, "structure_dictionaries"
            )
            if not os.path.exists(structures_folder):
                os.makedirs(structures_folder)
            transformed_folder = os.path.join(folder_preprocessed, "data_transformed")
            if not os.path.exists(transformed_folder):
                os.makedirs(transformed_folder)
            dual_graph_folder = os.path.join(folder_preprocessed, "xfg_dual")
            if not os.path.exists(dual_graph_folder):
                os.makedirs(dual_graph_folder)

            num_files = len(file_names)
            if isinstance(file_names, dict):
                file_names = list(file_names.values())
            for i, (preprocessed_file, file_name) in enumerate(
                zip(preprocessed_data, file_names)
            ):

                dual_graphs_filename = os.path.join(
                    dual_graph_folder, file_name[:-3] + ".p"
                )
                if not os.path.exists(dual_graphs_filename):

                    ####################################################################################################
                    # Build XFG (context graph)
                    print(
                        "\n--- Building graph for file : ",
                        file_name,
                        "(",
                        i,
                        "/",
                        num_files,
                        ")",
                    )

                    # Construct graph
                    try:
                        G, multi_edges = BuildContextualFlowGraph(
                            preprocessed_file,
                            functions_declared_in_files[i],
                            file_names[i],
                        )
                    except ValueError:
                        continue

                    # Print data to external file
                    print_graph_to_file(G, multi_edges, graph_folder, file_name)

                    ####################################################################################################
                    # XFG transformations (inline structures and abstract statements)

                    # Determine whether the data set has a specific structure pattern or not
                    specific_struct_name_pattern = get_data_characteristics(folder_raw)
                    if specific_struct_name_pattern:
                        rgx.struct_name = (
                            '%"[^"]*"'  # Adjust structure names in accordance
                        )

                    # Source code transformation: inline structure types
                    G, structures_dictionary = inline_struct_types(
                        G,
                        preprocessed_data_with_structure_def[i],
                        file_name,
                        specific_struct_name_pattern,
                    )

                    # Print structures dictionary
                    print_structure_dictionary(
                        structures_dictionary, structures_folder, file_name
                    )

                    # Source code transformation: abstract statement
                    G = abstract_statements_from_identifiers(G)

                    # Dump list of statements to be used in construct_vocabulary
                    stmt_list = [e[2]["stmt"] for e in G.edges(data=True)]
                    write_to = os.path.join(transformed_folder, file_name[:-3] + ".p")
                    print("Writing transformed data to", write_to)
                    i2v_utils.safe_pickle(stmt_list, write_to)

                    ####################################################################################################
                    # Build dual-XFG

                    dual_graphs_filename = os.path.join(
                        dual_graph_folder, file_name[:-3] + ".p"
                    )
                    if not os.path.exists(dual_graphs_filename):
                        print("Building dual graph for file ", file_name)

                        G_diff = disambiguate_stmts(G)
                        D = build_dual_graph(G_diff)  # dual-XFG
                        check_sanity(D, G)  # check the sanity of the produced graph

                        # Write dual graphs to file
                        print("Writing dual graphs to file ", dual_graphs_filename)
                        i2v_utils.safe_pickle(D, dual_graphs_filename)
                else:

                    print("--- Found dual-xfg for file : ", file_name, ", skipping...")

            ############################################################################################################
            # Write file indicating that the folder has been preprocessed
            f = open(data_preprocessing_done_filename, "w")
            f.close()

    return folders_raw

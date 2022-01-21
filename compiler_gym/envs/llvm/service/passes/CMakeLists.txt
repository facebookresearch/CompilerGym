# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

string(CONCAT _CMD
  "\"${Python3_EXECUTABLE}\" "
  "\"$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__extract_passes_from_llvm_source_tree,LOCATION>\""
  " \"${LLVM_SRC_DIR}/llvm\" > $@")
set_command_pythonpath(COMMAND "${_CMD}" RESULT _CMD)
cg_genrule(
  NAME passes_list
  OUTS "passes_list.csv"
  COMMAND ${_CMD}
  DEPENDS
    ::extract_passes_from_llvm_source_tree
)

string(CONCAT _CMD
  "\"${Python3_EXECUTABLE}\" "
  "\"$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__filter_action_space,LOCATION>\" < "
  " \"$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__passes_list,LOCATION>\" > $@")
set_command_pythonpath(COMMAND "${_CMD}" RESULT _CMD)
cg_genrule(
  NAME actions_csv
  OUTS "actions.csv"
  COMMAND ${_CMD}
  DEPENDS
    ::passes_list
    ::filter_action_space
)

cg_py_library(
  NAME
    common
  SRCS
    "common.py"
  PUBLIC
)

cg_py_library(
  NAME
    config
  SRCS
    "config.py"
  DEPS
    ::common
  PUBLIC
)

cg_py_binary(
  NAME
    extract_passes_from_llvm_source_tree
  SRCS
    "extract_passes_from_llvm_source_tree.py"
  DEPS
    ::common
    ::config
)

cg_py_binary(
  NAME
    make_action_space_genfiles
  SRCS
    "make_action_space_genfiles.py"
  DEPS
    ::common
    ::config
)

cg_py_binary(
  NAME
    filter_action_space
  SRCS
    "filter_action_space.py"
  DEPS
    ::common
    ::config
)

string(CONCAT _CMD
  "\"${Python3_EXECUTABLE}\" "
  "\"$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__make_action_space_genfiles,LOCATION>\""
  " $(@D) < \"$<TARGET_PROPERTY:compiler_gym__envs__llvm__service__passes__actions_csv,LOCATION>\""
)
set_command_pythonpath(COMMAND "${_CMD}" RESULT _CMD)
cg_genrule(
  NAME actions_genfiles
  OUTS
    "ActionEnum.h"
    "ActionSwitch.h"
    "ActionHeaders.h"
    "flags.txt"
    "flag_descriptions.txt"
  COMMAND
    ${_CMD}
  DEPENDS
    ::actions_csv
    ::make_action_space_genfiles
)

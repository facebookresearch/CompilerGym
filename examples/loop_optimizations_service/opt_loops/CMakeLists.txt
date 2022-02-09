# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()


llvm_map_components_to_libnames(_LLVM_LIBS analysis core irreader support passes)
cg_cc_binary(
  NAME
    opt_loops
  SRCS
    "opt_loops.cc"
  COPTS
    "-fno-rtti"
  ABS_DEPS
    ${_LLVM_LIBS}
    nlohmann_json::nlohmann_json
    compiler_gym::third_party::LLVM-Canon::IRCanonicalizer
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
)

ADD_CUSTOM_TARGET(link_opt_loops_target ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/examples/loop_optimizations_service/opt_loops/opt_loops ${CMAKE_SOURCE_DIR}/examples/loop_optimizations_service/opt_loops/opt_loops)

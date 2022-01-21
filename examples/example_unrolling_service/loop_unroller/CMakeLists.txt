# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()


llvm_map_components_to_libnames(_LLVM_LIBS analysis core irreader support passes)
cg_cc_binary(
  NAME
    loop_unroller
  SRCS
    "loop_unroller.cc"
  COPTS
    "-fno-rtti"
  ABS_DEPS
    ${_LLVM_LIBS}
  INCLUDES
    ${LLVM_INCLUDE_DIRS}
  DEFINES
    ${LLVM_DEFINITIONS}
)

ADD_CUSTOM_TARGET(link_loop_unroller_target ALL
                  COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/examples/example_unrolling_service/loop_unroller/loop_unroller ${CMAKE_SOURCE_DIR}/examples/example_unrolling_service/loop_unroller/loop_unroller)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

cg_filegroup(
  NAME "service"
  FILES
    "${CMAKE_CURRENT_LIST_DIR}/cgra_service.py"
    "${CMAKE_CURRENT_LIST_DIR}/compiler_gym-cgra-service"
    "${CMAKE_CURRENT_LIST_DIR}/compiler_gym-cgra-relative-placement-service"
)

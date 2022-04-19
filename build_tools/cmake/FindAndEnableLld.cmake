# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
if(
    CMAKE_EXE_LINKER_FLAGS_INIT
    OR CMAKE_MODULE_LINKER_FLAGS_INIT
    OR CMAKE_SHARED_LINKER_FLAGS_INIT
)
    return()
endif()

find_program(LLD ld.lld)
if(LLD)
    message("Found lld: ${LLD}")
    set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=${LLD} ${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS "-fuse-ld=${LLD} ${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=${LLD} ${CMAKE_SHARED_LINKER_FLAGS}")
else()
    message("lld not found")
endif()

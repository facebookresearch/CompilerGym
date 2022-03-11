# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_program(LLD lld)
if(LLD)
    message("Found lld: ${LLD}")
    set(CMAKE_EXE_LINKER_FLAGS_INIT "-fuse-ld=${LLD}")
    set(CMAKE_MODULE_LINKER_FLAGS_INIT "-fuse-ld=${LLD}")
    set(CMAKE_SHARED_LINKER_FLAGS_INIT "-fuse-ld=${LLD}")
else()
    message("lld not found")
endif()

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cg_add_all_subdirs()

if(COMPILER_GYM_ENABLE_LLVM_ENV)
    set(LEADERBOARD_DEPS ::llvm_instcount)
endif()
cg_py_library(
  NAME
    leaderboard
  SRCS
    "__init__.py"
  DEPS ${LEADERBOARD_DEPS}
  PUBLIC
)

if(COMPILER_GYM_ENABLE_LLVM_ENV)
    cg_py_library(
      NAME
        llvm_instcount
      SRCS
        "llvm_instcount.py"
      DEPS
        compiler_gym::bin::validate
        compiler_gym::compiler_env_state
        compiler_gym::envs::envs
        compiler_gym::util::util
      PUBLIC
    )
endif()

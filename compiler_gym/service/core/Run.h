// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "compiler_gym/service/core/CreateAndRunCompilerGymService.h"

namespace compiler_gym {

template <typename CompilationSession>
[[noreturn]] void createAndRunCompilerGymService(int* argc, char*** argv, const char* usage) {
  createAndRunCompilerGymServiceImpl<CompilationSession>(argc, argv, usage);
}

}  // namespace compiler_gym

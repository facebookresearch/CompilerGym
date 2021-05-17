// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "compiler_gym/service/runtime/CompilerGymService.h"
#include "compiler_gym/service/runtime/CreateAndRunCompilerGymServiceImpl.h"

namespace compiler_gym::runtime {

template <typename CompilationSessionType>
[[noreturn]] void createAndRunCompilerGymService(int argc, char** argv, const char* usage) {
  createAndRunCompilerGymServiceImpl<CompilationSessionType>(argc, argv, usage);
}

}  // namespace compiler_gym::runtime

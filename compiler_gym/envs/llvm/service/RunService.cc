// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmSession.h"
#include "compiler_gym/service/core/Core.h"
#include "compiler_gym/service/core/Run.h"

const char* usage = R"(LLVM CompilerGym service)";

using compiler_gym::createAndRunCompilerGymService;
using compiler_gym::llvm_service::LlvmSession;

int main(int argc, char** argv) { createAndRunCompilerGymService<LlvmSession>(argc, argv, usage); }

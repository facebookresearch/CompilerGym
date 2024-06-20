// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmServiceContext.h"
#include "compiler_gym/envs/llvm/service/LlvmSession.h"
#include "compiler_gym/service/runtime/Runtime.h"

const char* usage = R"(LLVM CompilerGym service)";

using namespace compiler_gym::runtime;
using namespace compiler_gym::llvm_service;

int main(int argc, char** argv) {
  return createAndRunCompilerGymService<LlvmSession, LlvmServiceContext>(argc, argv, usage);
}

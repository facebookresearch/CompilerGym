// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/util/RunService.h"

#include "compiler_gym/envs/llvm/service/LlvmService.h"

const char* usage = R"(LLVM CompilerGym service)";

using namespace compiler_gym::util;
using namespace compiler_gym::llvm_service;

int main(int argc, char** argv) { return runService<LlvmService>(&argc, &argv, usage); }

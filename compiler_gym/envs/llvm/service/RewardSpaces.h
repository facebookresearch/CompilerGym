// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <vector>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::llvm_service {

// The available LLVM reward spaces.
//
// Housekeeping rules - to add a new reward space:
//   1. Add a new entry to this LlvmRewardSpace enum.
//   2. Add a new switch case to intToLlvmRewardSpace().
//   3. Add a new switch case to LlvmEnvironment::getReward().
//   4. Run `bazel test //compiler_gym/...` and update the newly failing tests.
enum class LlvmRewardSpace {
  // Returns the number of IR instructions in the current module.
  IR_INSTRUCTION_COUNT,
  IR_INSTRUCTION_COUNT_O3,
  IR_INSTRUCTION_COUNT_Oz,
  OBJECT_TEXT_SIZE_BYTES,
  OBJECT_TEXT_SIZE_O3,
  OBJECT_TEXT_SIZE_Oz,
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
  TEXT_SIZE_BYTES,
  TEXT_SIZE_O3,
  TEXT_SIZE_Oz,
#endif
};

// Get the list of available reward spaces.
std::vector<RewardSpace> getLlvmRewardSpaceList();

}  // namespace compiler_gym::llvm_service

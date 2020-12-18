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
  // Returns a reward in the range (0,inf) that is the ratio of the number of
  // LLVM instructions in the current module, relative to the number of
  // instructions when optimized with -Oz. This is quick to evaluate as it does
  // not require compiling the module.
  IR_INSTRUCTION_COUNT,
  IR_INSTRUCTION_COUNT_OZ,
  IR_INSTRUCTION_COUNT_O3,
  // Same as above, but the reward at a given timestep is change in
  // instantaneous reward relative to the previous step. i.e:
  //     R^diff_{t} = R_{t} - R_{t-1}.
  IR_INSTRUCTION_COUNT_OZ_DIFF,
};

// Get the list of available reward spaces.
std::vector<RewardSpace> getLlvmRewardSpaceList();

}  // namespace compiler_gym::llvm_service

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <magic_enum.hpp>
#include <optional>

#include "compiler_gym/envs/llvm/service/RewardSpaces.h"
#include "llvm/IR/Module.h"

namespace compiler_gym::llvm_service {

using BaselineCosts = std::array<std::optional<double>, magic_enum::enum_count<LlvmRewardSpace>()>;

// Compute the cost using a given reward space. A lower cost is better.
double getCost(const LlvmRewardSpace& space, llvm::Module& module);

// Compute the baseline costs used for calculating reward.
void setbaselineCosts(const llvm::Module& unoptimizedModule, BaselineCosts* baselineCosts);

}  // namespace compiler_gym::llvm_service

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <magic_enum.hpp>
#include <optional>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/RewardSpaces.h"
#include "llvm/IR/Module.h"

namespace compiler_gym::llvm_service {

enum class LlvmCostFunction {
  // The number of instructions in the LLVM-IR module. This is fast to compute
  // and deterministic.
  IR_INSTRUCTION_COUNT,
  // Returns the size (in bytes) of the .TEXT section of the compiled module.
  NATIVE_TEXT_SIZE_BYTES,
};

enum class LlvmBaselinePolicy {
  O0,  // No optimizations.
  O3,  // -O3 optimizations.
  Oz,  // -Oz optimizations.
};

constexpr size_t numCosts = magic_enum::enum_count<LlvmCostFunction>();
constexpr size_t numBaselineCosts = magic_enum::enum_count<LlvmBaselinePolicy>() * numCosts;

using BaselineCosts = std::array<double, numBaselineCosts>;
using PreviousCosts = std::array<std::optional<double>, numCosts>;

// TODO(cummins): Refactor cost calculation to allow graceful error handling
// by returning a grpc::Status.

// Compute the cost using a given cost function. A lower cost is better.
double getCost(const LlvmCostFunction& cost, llvm::Module& module,
               const boost::filesystem::path& workingDirectory);

// Return a baseline cost.
double getBaselineCost(const BaselineCosts& baselineCosts, LlvmBaselinePolicy policy,
                       LlvmCostFunction cost);

// Compute the costs of baseline policies.
void setbaselineCosts(const llvm::Module& unoptimizedModule, BaselineCosts* baselineCosts,
                      const boost::filesystem::path& workingDirectory);

// Translate from reward space to a cost function.
LlvmCostFunction getCostFunction(LlvmRewardSpace space);

// Translate from reward space to a baseline policy.
LlvmBaselinePolicy getBaselinePolicy(LlvmRewardSpace space);

}  // namespace compiler_gym::llvm_service

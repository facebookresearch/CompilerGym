// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <magic_enum.hpp>
#include <optional>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/BenchmarkDynamicConfig.h"
#include "llvm/IR/Module.h"

namespace compiler_gym::llvm_service {

/**
 * A cost function for LLVM benchmarks.
 */
enum class LlvmCostFunction {
  /**
   * The number of instructions in the LLVM-IR module.
   *
   * IR instruction count is fast to compute and deterministic.
   */
  IR_INSTRUCTION_COUNT,
  /**
   * Returns the size (in bytes) of the .TEXT section of the compiled module.
   */
  OBJECT_TEXT_SIZE_BYTES,
  /**
   * Returns the size (in bytes) of the .TEXT section of the compiled binary.
   */
  TEXT_SIZE_BYTES,
};

/**
 * LLVM's builtin policies.
 */
enum class LlvmBaselinePolicy {
  O0,  ///< No optimizations.
  O3,  ///< `-O3` optimizations.
  Oz,  ///< `-Oz` optimizations.
};

constexpr size_t numCosts = magic_enum::enum_count<LlvmCostFunction>();
constexpr size_t numBaselineCosts = magic_enum::enum_count<LlvmBaselinePolicy>() * numCosts;

using BaselineCosts = std::array<double, numBaselineCosts>;
using PreviousCosts = std::array<std::optional<double>, numCosts>;

/**
 * Apply the given baseline optimizations.
 *
 * @param module The module to optimize.
 * @param optLevel The runtime optimization level.
 * @param sizeLevel The size optimization level
 * @return Whether the baseline optimizations modified the module.
 */
bool applyBaselineOptimizationsToModule(llvm::Module* module, unsigned optLevel,
                                        unsigned sizeLevel);

/**
 * Compute the cost using a given cost function. A lower cost is better.
 *
 * @param costFunction The cost function to use.
 * @param module The module to compute the cost for.
 * @param workingDirectory A directory that can be used for temporary file
 *    storage.
 * @param cost The cost to write.
 * @return `OK` on success.
 */
[[nodiscard]] grpc::Status setCost(const LlvmCostFunction& costFunction, llvm::Module& module,
                                   const boost::filesystem::path& workingDirectory,
                                   const BenchmarkDynamicConfig& dynamicConfig, double* cost);

/**
 * Return a baseline cost.
 *
 * @param baselineCosts The baseline costs list.
 * @param policy The baseline policy to return the cost of.
 * @param cost The cost function to use.
 * @return A cost.
 */
double getBaselineCost(const BaselineCosts& baselineCosts, LlvmBaselinePolicy policy,
                       LlvmCostFunction cost);

/**
 * Compute the costs of baseline policies.
 *
 * \note The `unoptimizedModule` parameter is unmodified, but is not const
 *    because various LLVM API calls require a mutable reference.
 *
 * @param unoptimizedModule The module to compute the baseline costs of.
 * @param baselineCosts The costs to write.
 * @param workingDirectory A directory that can be used for temporary file
 *    storage.
 */
[[nodiscard]] grpc::Status setBaselineCosts(llvm::Module& unoptimizedModule,
                                            const boost::filesystem::path& workingDirectory,
                                            const BenchmarkDynamicConfig& dynamicConfig,
                                            BaselineCosts* baselineCosts);

}  // namespace compiler_gym::llvm_service

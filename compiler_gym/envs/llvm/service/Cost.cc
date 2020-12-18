// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/Cost.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace compiler_gym::llvm_service {

namespace {

// Apply the optimizations from a given LLVM optimization level.
bool applyBaselineOptimizations(llvm::Module* module, unsigned optLevel, unsigned sizeLevel) {
  llvm::legacy::PassManager passManager;
  llvm::legacy::FunctionPassManager functionPassManager(module);

  llvm::PassManagerBuilder builder;
  builder.OptLevel = optLevel;
  builder.SizeLevel = sizeLevel;
  if (optLevel > 1) {
    builder.Inliner = llvm::createFunctionInliningPass(optLevel, sizeLevel, false);
  }

  builder.populateFunctionPassManager(functionPassManager);
  builder.populateModulePassManager(passManager);

  bool changed = passManager.run(*module);
  changed |= (functionPassManager.doInitialization() ? 1 : 0);
  for (auto& function : *module) {
    changed |= (functionPassManager.run(function) ? 1 : 0);
  }
  changed |= (functionPassManager.doFinalization() ? 1 : 0);

  return changed;
}

}  // anonymous namespace

double getCost(const LlvmRewardSpace& space, llvm::Module& module) {
  switch (space) {
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT:
      return -static_cast<double>(module.getInstructionCount());
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT_OZ:
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT_OZ_DIFF:
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT_O3:
      // A module should never contain zero instructions, but this isn't
      // enforced. Instead, clamp the minimum instruction count to 1 to prevent
      // divide-by-zero errors when calculating the ratio of costs.
      return std::max(static_cast<double>(module.getInstructionCount()), 1.0);
  }
}

void setbaselineCosts(const llvm::Module& unoptimizedModule, BaselineCosts* baselineCosts) {
  // Create a copy of the unoptimized module and apply the default set of LLVM
  // optimizations.
  std::unique_ptr<llvm::Module> moduleOz = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleOz.get(), /*optLevel=*/2, /*sizeLevel=*/2);

  std::unique_ptr<llvm::Module> moduleO3 = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleO3.get(), /*optLevel=*/3, /*sizeLevel=*/0);

  for (const auto space : magic_enum::enum_values<LlvmRewardSpace>()) {
    switch (space) {
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT:
        (*baselineCosts)[static_cast<size_t>(space)].reset();
        break;
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_OZ:
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_OZ_DIFF:
        (*baselineCosts)[static_cast<size_t>(space)] = getCost(space, *moduleOz);
        break;
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_O3:
        (*baselineCosts)[static_cast<size_t>(space)] = getCost(space, *moduleO3);
        break;
    }
  }
}

}  // namespace compiler_gym::llvm_service

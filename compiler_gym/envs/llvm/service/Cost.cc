// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/Cost.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <subprocess/subprocess.hpp>
#include <system_error>

#include "boost/filesystem.hpp"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace fs = boost::filesystem;
using grpc::Status;
using grpc::StatusCode;

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

// Serialize the module to a string.
std::string moduleToString(llvm::Module& module) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  module.print(rso, /*AAW=*/nullptr);
  return str;
}

Status getNativeTextSizeInBytes(llvm::Module& module, int64_t* value,
                                const fs::path& workingDirectory) {
  const auto clangPath = util::getRunfilesPath("compiler_gym/third_party/llvm/clang");
  DCHECK(fs::exists(clangPath)) << "File not found: " << clangPath.string();

  const auto tmpFile = fs::unique_path(workingDirectory / "module-%%%%.o");

  // Lower the module to an object file using clang.
  // TODO(cummins): Use clang driver APIs rather than invoking command line.
  // https://github.com/llvm/llvm-project/blob/b8d2b6f6cf6015751fc950c3e8149404e8b37fe8/clang/examples/clang-interpreter/main.cpp
  const std::string ir = moduleToString(module);
  auto clang =
      subprocess::Popen({clangPath.string(), "-O0", "-xir", "-", "-c", "-o", tmpFile.string()},
                        subprocess::input{subprocess::PIPE}, subprocess::output{subprocess::PIPE},
                        subprocess::error{subprocess::PIPE});
  const auto clangOutput = clang.communicate(ir.c_str(), ir.size());
  if (clang.retcode()) {
    fs::remove(tmpFile);
    std::string error(clangOutput.second.buf.begin(), clangOutput.second.buf.end());
    return Status(StatusCode::INTERNAL, error);
  }

  // Use size to measure the size of the text section.
  // NOTE(cummins): Requires that `awk` and `size` are in PATH.
  auto size =
      subprocess::Popen(fmt::format("size {} | awk 'NR==2 {}'", tmpFile.string(), "{print $1}"),
                        subprocess::shell{true}, subprocess::output{subprocess::PIPE},
                        subprocess::error{subprocess::PIPE});
  const auto sizeOutput = size.communicate();
  fs::remove(tmpFile);
  if (size.retcode()) {
    std::string error(sizeOutput.second.buf.begin(), sizeOutput.second.buf.end());
    return Status(StatusCode::INTERNAL, error);
  }

  std::string stdout{sizeOutput.first.buf.begin(), sizeOutput.first.buf.end()};
  try {
    *value = std::stoi(stdout);
  } catch (std::exception const& e) {
    return Status(StatusCode::INTERNAL, fmt::format("Failed to parse integer: `{}`\n", stdout));
  }
  return Status::OK;
}

inline size_t getBaselineCostIndex(LlvmBaselinePolicy policy, LlvmCostFunction cost) {
  return static_cast<size_t>(magic_enum::enum_count<LlvmCostFunction>()) *
             static_cast<size_t>(policy) +
         static_cast<size_t>(cost);
}

}  // anonymous namespace

double getCost(const LlvmCostFunction& cost, llvm::Module& module,
               const fs::path& workingDirectory) {
  switch (cost) {
    case LlvmCostFunction::IR_INSTRUCTION_COUNT:
      return static_cast<double>(module.getInstructionCount());
    case LlvmCostFunction::NATIVE_TEXT_SIZE_BYTES: {
      int64_t size;
      const auto status = getNativeTextSizeInBytes(module, &size, workingDirectory);
      CHECK(status.ok()) << status.error_message();
      return static_cast<double>(size);
    }
  }
}

double getBaselineCost(const BaselineCosts& baselineCosts, LlvmBaselinePolicy policy,
                       LlvmCostFunction cost) {
  return baselineCosts[getBaselineCostIndex(policy, cost)];
}

void setbaselineCosts(const llvm::Module& unoptimizedModule, BaselineCosts* baselineCosts,
                      const fs::path& workingDirectory) {
  // Create a copy of the unoptimized module and apply the default set of LLVM
  // optimizations.
  std::unique_ptr<llvm::Module> moduleO0 = llvm::CloneModule(unoptimizedModule);

  std::unique_ptr<llvm::Module> moduleOz = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleOz.get(), /*optLevel=*/2, /*sizeLevel=*/2);

  std::unique_ptr<llvm::Module> moduleO3 = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleO3.get(), /*optLevel=*/3, /*sizeLevel=*/0);

  for (const auto policy : magic_enum::enum_values<LlvmBaselinePolicy>()) {
    // Set the baseline module.
    llvm::Module* baselineModule{nullptr};
    switch (policy) {
      case LlvmBaselinePolicy::O0:
        baselineModule = moduleO0.get();
        break;
      case LlvmBaselinePolicy::O3:
        baselineModule = moduleO3.get();
        break;
      case LlvmBaselinePolicy::Oz:
        baselineModule = moduleOz.get();
        break;
    }
    DCHECK(baselineModule);

    // Compute and set the baseline costs.
    for (const auto cost : magic_enum::enum_values<LlvmCostFunction>()) {
      const auto idx = getBaselineCostIndex(policy, cost);
      const auto cc = getCost(cost, *baselineModule, workingDirectory);
      (*baselineCosts)[idx] = cc;
    }
  }
}

LlvmCostFunction getCostFunction(LlvmRewardSpace space) {
  switch (space) {
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT:
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT_O3:
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT_Oz:
      return LlvmCostFunction::IR_INSTRUCTION_COUNT;
    case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES:
    case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES_O3:
    case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES_Oz:
      return LlvmCostFunction::NATIVE_TEXT_SIZE_BYTES;
  }
}

LlvmBaselinePolicy getBaselinePolicy(LlvmRewardSpace space) {
  switch (space) {
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT:
    case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES:
      return LlvmBaselinePolicy::O0;
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT_O3:
    case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES_O3:
      return LlvmBaselinePolicy::O3;
    case LlvmRewardSpace::IR_INSTRUCTION_COUNT_Oz:
    case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES_Oz:
      return LlvmBaselinePolicy::Oz;
  }
}

}  // namespace compiler_gym::llvm_service

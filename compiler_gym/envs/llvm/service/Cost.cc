// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/Cost.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <boost/asio.hpp>
#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <future>
#include <system_error>

#include "boost/filesystem.hpp"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/Subprocess.h"
#include "compiler_gym/util/Unreachable.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace fs = boost::filesystem;
namespace bp = boost::process;
using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym::llvm_service {

namespace {

// Serialize the module to a string.
std::string moduleToString(llvm::Module& module) {
  std::string str;
  llvm::raw_string_ostream rso(str);
  module.print(rso, /*AAW=*/nullptr);
  return str;
}

Status writeBitcodeFile(const llvm::Module& module, const fs::path& path) {
  std::error_code error;
  llvm::raw_fd_ostream outfile(path.string(), error);
  if (error.value()) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to write bitcode file: {}", path.string()));
  }
  llvm::WriteBitcodeToFile(module, outfile);
  return Status::OK;
}

util::LocalShellCommand getBuildCommand(const BenchmarkDynamicConfig& dynamicConfig,
                                        bool compile_only) {
  // Append the '-c' flag to compile-only jobs.
  if (compile_only) {
    Command newCommand;
    newCommand.CopyFrom(dynamicConfig.buildCommand().proto());

    // Determine if the compilation specifies an output file using the `-o
    // <file>` flag. If not, then add one so that when we append the `-c` flag
    // we still know where the generated object file will go.
    bool outfileSpecified = false;
    for (int i = 0; i < newCommand.argument_size() - 1; ++i) {
      if (newCommand.argument(i) == "-o") {
        outfileSpecified = true;
        break;
      }
    }
    if (!outfileSpecified) {
      newCommand.add_argument("-o");
      if (newCommand.outfile_size() < 1) {
        newCommand.add_argument("a.out");
        newCommand.add_outfile("a.out");
      } else {
        const auto& outfile = newCommand.outfile(0);
        newCommand.add_argument(outfile);
      }
    }

    newCommand.add_argument("-c");

    return util::LocalShellCommand(newCommand);
  }
  return dynamicConfig.buildCommand();
}

inline size_t getBaselineCostIndex(LlvmBaselinePolicy policy, LlvmCostFunction cost) {
  return static_cast<size_t>(magic_enum::enum_count<LlvmCostFunction>()) *
             static_cast<size_t>(policy) +
         static_cast<size_t>(cost);
}

}  // anonymous namespace

/**
 * Apply the given baseline optimizations.
 *
 * @param module The module to optimize.
 * @param optLevel The runtime optimization level.
 * @param sizeLevel The size optimization level
 * @return Whether the baseline optimizations modified the module.
 */
bool applyBaselineOptimizationsToModule(llvm::Module* module, unsigned optLevel,
                                        unsigned sizeLevel) {
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

Status setCost(const LlvmCostFunction& costFunction, llvm::Module& module,
               const fs::path& workingDirectory, const BenchmarkDynamicConfig& dynamicConfig,
               double* cost) {
  switch (costFunction) {
    case LlvmCostFunction::IR_INSTRUCTION_COUNT: {
      *cost = static_cast<double>(module.getInstructionCount());
      break;
    }
    case LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES: {
      *cost = static_cast<double>(1);
      break;
    }
    case LlvmCostFunction::TEXT_SIZE_BYTES: {
      *cost = static_cast<double>(1);
      break;
    }
    default:
      UNREACHABLE(fmt::format("Unhandled cost function: {}", costFunction));
  }
  return Status::OK;
}

double getBaselineCost(const BaselineCosts& baselineCosts, LlvmBaselinePolicy policy,
                       LlvmCostFunction cost) {
  return baselineCosts[getBaselineCostIndex(policy, cost)];
}

Status setBaselineCosts(llvm::Module& unoptimizedModule, const fs::path& workingDirectory,
                        const BenchmarkDynamicConfig& dynamicConfig, BaselineCosts* baselineCosts) {
  llvm::Module* moduleO0 = &unoptimizedModule;

  // Create a copy of the unoptimized module and apply the default set of LLVM
  // optimizations.
  std::unique_ptr<llvm::Module> moduleOz = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizationsToModule(moduleOz.get(), /*optLevel=*/2, /*sizeLevel=*/2);

  std::unique_ptr<llvm::Module> moduleO3 = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizationsToModule(moduleO3.get(), /*optLevel=*/3, /*sizeLevel=*/0);

  for (const auto policy : magic_enum::enum_values<LlvmBaselinePolicy>()) {
    // Set the baseline module.
    llvm::Module* baselineModule{nullptr};
    switch (policy) {
      case LlvmBaselinePolicy::O0:
        baselineModule = moduleO0;
        break;
      case LlvmBaselinePolicy::O3:
        baselineModule = moduleO3.get();
        break;
      case LlvmBaselinePolicy::Oz:
        baselineModule = moduleOz.get();
        break;
      default:
        UNREACHABLE("Unknown baseline policy");
    }
    DCHECK(baselineModule);

    // Compute and set the baseline costs.
    for (const auto cost : magic_enum::enum_values<LlvmCostFunction>()) {
      const auto idx = getBaselineCostIndex(policy, cost);
      RETURN_IF_ERROR(
          setCost(cost, *baselineModule, workingDirectory, dynamicConfig, &(*baselineCosts)[idx]));
    }
  }

  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

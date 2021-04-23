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
#include "compiler_gym/util/Unreachable.h"
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

// For the experimental binary .text size cost, getTextSizeInBytes() is extended
// to support a list of additional args to pass to clang.
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
Status getTextSizeInBytes(llvm::Module& module, int64_t* value,
                          const std::vector<std::string>& clangArgs,
                          const fs::path& workingDirectory) {
#else
Status getTextSizeInBytes(llvm::Module& module, int64_t* value, const fs::path& workingDirectory) {
#endif
  const auto clangPath = util::getSiteDataPath("llvm-v0/bin/clang");
  const auto llvmSizePath = util::getSiteDataPath("llvm-v0/bin/llvm-size");
  DCHECK(fs::exists(clangPath)) << fmt::format("File not found: {}", clangPath.string());
  DCHECK(fs::exists(llvmSizePath)) << fmt::format("File not found: {}", llvmSizePath.string());

  // Lower the module to an object file using clang and extract the .text
  // section size using llvm-size.
  const std::string ir = moduleToString(module);

  const auto tmpFile = fs::unique_path(workingDirectory / "obj-%%%%");

#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
  std::vector<std::string> clangCmd{clangPath.string(), "-w", "-xir", "-", "-o", tmpFile.string()};
  clangCmd.insert(clangCmd.end(), clangArgs.begin(), clangArgs.end());
  auto clang =
      subprocess::Popen(clangCmd, subprocess::input{subprocess::PIPE},
                        subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});
#else
  auto clang =
      subprocess::Popen({clangPath.string(), "-w", "-xir", "-", "-o", tmpFile.string(), "-c"},
                        subprocess::input{subprocess::PIPE}, subprocess::output{subprocess::PIPE},
                        subprocess::error{subprocess::PIPE});
#endif
  const auto clangOutput = clang.communicate(ir.c_str(), ir.size());
  if (clang.retcode()) {
    fs::remove(tmpFile);
    const std::string error(clangOutput.second.buf.begin(), clangOutput.second.buf.end());
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("Failed to compute .text size cost.\nError from clang:\n{}", error));
  }

  auto llvmSize =
      subprocess::Popen({llvmSizePath.string(), tmpFile.string()},
                        subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});
  const auto sizeOutput = llvmSize.communicate();
  fs::remove(tmpFile);
  if (llvmSize.retcode()) {
    const std::string error(sizeOutput.second.buf.begin(), sizeOutput.second.buf.end());
    return Status(
        StatusCode::INVALID_ARGUMENT,
        fmt::format("Failed to compute .text size cost.\nError from llvm-size:\n{}", error));
  }

  // The output of llvm-size is in berkley format, e.g.:
  //
  //     $ llvm-size foo.o
  //     __TEXT __DATA __OBJC others dec hex
  //     127    0      0      32	   159 9f
  //
  // Skip the first line of output and read an integer from the start of the
  // second line:
  const std::string stdout{sizeOutput.first.buf.begin(), sizeOutput.first.buf.end()};
  const size_t eol = stdout.find('\n');
  const size_t tab = stdout.find('\t', eol + 1);
  if (eol == std::string::npos || tab == std::string::npos) {
    return Status(StatusCode::INTERNAL, fmt::format("Failed to parse .TEXT size: `{}`\n", stdout));
  }
  const std::string extracted = stdout.substr(eol, tab - eol);
  try {
    *value = std::stoi(extracted);
  } catch (std::exception const& e) {
    return Status(StatusCode::INTERNAL, fmt::format("Failed to parse .TEXT size: `{}`\n", stdout));
  }
  return Status::OK;
}

inline size_t getBaselineCostIndex(LlvmBaselinePolicy policy, LlvmCostFunction cost) {
  return static_cast<size_t>(magic_enum::enum_count<LlvmCostFunction>()) *
             static_cast<size_t>(policy) +
         static_cast<size_t>(cost);
}

}  // anonymous namespace

Status setCost(const LlvmCostFunction& costFunction, llvm::Module& module,
               const fs::path& workingDirectory, double* cost) {
  switch (costFunction) {
    case LlvmCostFunction::IR_INSTRUCTION_COUNT: {
      *cost = static_cast<double>(module.getInstructionCount());
      break;
    }
    case LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES: {
      int64_t size;
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
      RETURN_IF_ERROR(getTextSizeInBytes(module, &size, {"-c"}, workingDirectory));
#else
      RETURN_IF_ERROR(getTextSizeInBytes(module, &size, workingDirectory));
#endif
      *cost = static_cast<double>(size);
      break;
    }
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST

#ifdef __APPLE__
#define SYSTEM_LIBRARIES \
  "-L"                   \
  "/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib"
#else
#define SYSTEM_LIBRARIES
#endif
    case LlvmCostFunction::TEXT_SIZE_BYTES: {
      int64_t size;
      RETURN_IF_ERROR(getTextSizeInBytes(module, &size, {SYSTEM_LIBRARIES}, workingDirectory));
      *cost = static_cast<double>(size);
      break;
    }
#endif
    default:
      UNREACHABLE(fmt::format("Unhandled cost function: {}", costFunction));
  }
  return Status::OK;
}

double getBaselineCost(const BaselineCosts& baselineCosts, LlvmBaselinePolicy policy,
                       LlvmCostFunction cost) {
  return baselineCosts[getBaselineCostIndex(policy, cost)];
}

Status setBaselineCosts(llvm::Module& unoptimizedModule, BaselineCosts* baselineCosts,
                        const fs::path& workingDirectory) {
  llvm::Module* moduleO0 = &unoptimizedModule;

  // Create a copy of the unoptimized module and apply the default set of LLVM
  // optimizations.
  std::unique_ptr<llvm::Module> moduleOz = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleOz.get(), /*optLevel=*/2, /*sizeLevel=*/2);

  std::unique_ptr<llvm::Module> moduleO3 = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleO3.get(), /*optLevel=*/3, /*sizeLevel=*/0);

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
      RETURN_IF_ERROR(setCost(cost, *baselineModule, workingDirectory, &(*baselineCosts)[idx]));
    }
  }

  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

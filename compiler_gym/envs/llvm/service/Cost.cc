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

Status runCommandOnModule(const std::string& cmd, llvm::Module& module, std::string* stdout) {
  const std::string ir = moduleToString(module);
  VLOG(4) << "$ " << cmd;
  auto process =
      subprocess::Popen(cmd, subprocess::shell{true}, subprocess::input{subprocess::PIPE},
                        subprocess::output{subprocess::PIPE}, subprocess::error{subprocess::PIPE});
  const auto output = process.communicate(ir.c_str(), ir.size());
  if (process.retcode()) {
    std::string error(output.second.buf.begin(), output.second.buf.end());
    return Status(StatusCode::INTERNAL, error);
  }
  *stdout = std::string(output.first.buf.begin(), output.first.buf.end());
  return Status::OK;
}

Status getNativeTextSizeInBytes(llvm::Module& module, int64_t* value,
                                const fs::path& workingDirectory) {
  const auto clang = util::getRunfilesPath("CompilerGym/compiler_gym/third_party/llvm/clang");
  DCHECK(fs::exists(clang)) << "File not found: " << clang.string();
  const auto tmpFile = fs::unique_path(workingDirectory / "module-%%%%.o");

  // NOTE(cummins): Requires awk and size being in the path.
  const std::string cmd =
      fmt::format("set -o pipefail; {} -O0 -xir - -c -o {} && size {} | awk 'NR==2 {}'",
                  clang.string(), tmpFile.string(), tmpFile.string(), "{print $1}");
  std::string stdout;
  auto status = runCommandOnModule(cmd, module, &stdout);

  fs::remove(tmpFile);
  RETURN_IF_ERROR(status);

  try {
    *value = std::stoi(stdout);
  } catch (std::exception const& e) {
    return Status(StatusCode::INTERNAL, fmt::format("Failed to read command output as integer.\n"
                                                    "Command: {}\n"
                                                    "Stdout: {}\n",
                                                    cmd, stdout));
  }
  return Status::OK;
}

}  // anonymous namespace

double getCost(const LlvmRewardSpace& space, llvm::Module& module,
               const fs::path& workingDirectory) {
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
    case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES: {
      int64_t size;
      CHECK(getNativeTextSizeInBytes(module, &size, workingDirectory).ok());
      return -static_cast<double>(size);
    }
  }
}

void setbaselineCosts(const llvm::Module& unoptimizedModule, BaselineCosts* baselineCosts,
                      const fs::path& workingDirectory) {
  // Create a copy of the unoptimized module and apply the default set of LLVM
  // optimizations.
  std::unique_ptr<llvm::Module> moduleOz = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleOz.get(), /*optLevel=*/2, /*sizeLevel=*/2);

  std::unique_ptr<llvm::Module> moduleO3 = llvm::CloneModule(unoptimizedModule);
  applyBaselineOptimizations(moduleO3.get(), /*optLevel=*/3, /*sizeLevel=*/0);

  for (const auto space : magic_enum::enum_values<LlvmRewardSpace>()) {
    switch (space) {
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT:
      case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES:
        (*baselineCosts)[static_cast<size_t>(space)].reset();
        break;
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_OZ:
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_OZ_DIFF:
        (*baselineCosts)[static_cast<size_t>(space)] = getCost(space, *moduleOz, workingDirectory);
        break;
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_O3:
        (*baselineCosts)[static_cast<size_t>(space)] = getCost(space, *moduleO3, workingDirectory);
        break;
    }
  }
}

}  // namespace compiler_gym::llvm_service

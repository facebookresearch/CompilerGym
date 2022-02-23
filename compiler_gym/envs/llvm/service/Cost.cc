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

Status getTextSizeInBytes(const fs::path& file, int64_t* value) {
  const auto llvmSizePath = util::getSiteDataPath("llvm-v0/bin/llvm-size");
  DCHECK(fs::exists(llvmSizePath)) << fmt::format("File not found: {}", llvmSizePath.string());

  const std::string llvmSizeCmd = fmt::format("{} {}", llvmSizePath.string(), file.string());

  boost::asio::io_context llvmSizeStdoutStream;
  std::future<std::string> llvmSizeStdoutFuture;
  std::string llvmSizeOutput;

  try {
    bp::child llvmSize(llvmSizeCmd, bp::std_in.close(), bp::std_out > llvmSizeStdoutFuture,
                       bp::std_err > bp::null, llvmSizeStdoutStream);

    llvmSizeStdoutStream.run_for(std::chrono::seconds(60));
    if (llvmSizeStdoutStream.poll()) {
      return Status(StatusCode::DEADLINE_EXCEEDED,
                    fmt::format("Failed to compute .text size cost within 60 seconds"));
    }
    llvmSize.wait();
    llvmSizeOutput = llvmSizeStdoutFuture.get();

    if (llvmSize.exit_code()) {
      return Status(StatusCode::INVALID_ARGUMENT, fmt::format("Failed to compute .text size cost. "
                                                              "Command returned exit code {}: {}",
                                                              llvmSize.exit_code(), llvmSizeCmd));
    }
  } catch (bp::process_error& e) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("Failed to compute .text size cost: {}", e.what()));
  }

  // The output of llvm-size is in berkley format, e.g.:
  //
  //     $ llvm-size foo.o
  //     __TEXT __DATA __OBJC others dec hex
  //     127    0      0      32	   159 9f
  //
  // Skip the first line of output and read an integer from the start of the
  // second line:
  const size_t eol = llvmSizeOutput.find('\n');
  const size_t tab = llvmSizeOutput.find('\t', eol + 1);
  if (eol == std::string::npos || tab == std::string::npos) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to parse .TEXT size: `{}`\n", llvmSizeOutput));
  }
  const std::string extracted = llvmSizeOutput.substr(eol, tab - eol);
  try {
    *value = std::stoi(extracted);
  } catch (std::exception const& e) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to parse .TEXT size: `{}`\n", llvmSizeOutput));
  }

  return Status::OK;
}

#define TEXT_SIZE_RETURN_IF_ERROR(expr)                                                            \
  while (1) {                                                                                      \
    const auto status = expr;                                                                      \
    if (!status.ok()) {                                                                            \
      return Status(StatusCode::INVALID_ARGUMENT,                                                  \
                    fmt::format("Failed to compute .text size cost: {}", status.error_message())); \
    }                                                                                              \
    break;                                                                                         \
  }

Status getTextSizeInBytes(llvm::Module& module, int64_t* value, const fs::path& workingDirectory,
                          const util::LocalShellCommand& buildCommand) {
  if (buildCommand.outfiles().size() != 1) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("Expected a single output from build job, actual: {}. Command: {}",
                              buildCommand.outfiles().size(), buildCommand.commandline()));
  }
  const auto& outfile = buildCommand.outfiles()[0];

  // Write the bitcode to the expected place.
  RETURN_IF_ERROR(writeBitcodeFile(module, "out.bc"));

  TEXT_SIZE_RETURN_IF_ERROR(buildCommand.checkInfiles());
  TEXT_SIZE_RETURN_IF_ERROR(buildCommand.checkCall());
  TEXT_SIZE_RETURN_IF_ERROR(buildCommand.checkOutfiles());

  return getTextSizeInBytes(outfile, value);
}

#undef TEXT_SIZE_RETURN_IF_ERROR

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

Status getTextSizeInBytes(llvm::Module& module, int64_t* value, const fs::path& workingDirectory,
                          const BenchmarkDynamicConfig& dynamicConfig, bool compile_only = true) {
  if (dynamicConfig.isBuildable()) {
    if (chdir(dynamicConfig.scratchDirectory().string().c_str())) {
      return Status(StatusCode::INTERNAL, fmt::format("Failed to set working directory: {}",
                                                      dynamicConfig.scratchDirectory().string()));
    }

    const util::LocalShellCommand buildCommand = getBuildCommand(dynamicConfig, compile_only);
    return getTextSizeInBytes(module, value, workingDirectory, buildCommand);
  }

  // TODO(cummins): We only provide an implementation for the compile_only path
  // here.

  const auto clangPath = util::getSiteDataPath("llvm-v0/bin/clang");
  const auto llvmSizePath = util::getSiteDataPath("llvm-v0/bin/llvm-size");
  DCHECK(fs::exists(clangPath)) << fmt::format("File not found: {}", clangPath.string());
  DCHECK(fs::exists(llvmSizePath)) << fmt::format("File not found: {}", llvmSizePath.string());

  // Lower the module to an object file using clang and extract the .text
  // section size using llvm-size.
  const std::string ir = moduleToString(module);

  const auto tmpFile = fs::unique_path(workingDirectory / "obj-%%%%.o");

  try {
    const std::string clangCmd =
        fmt::format("{} -w -xir - -o {} -c", clangPath.string(), tmpFile.string());

    boost::asio::io_context clangContext;
    auto stdinBuffer{boost::asio::buffer(ir)};
    bp::async_pipe stdinPipe(clangContext);
    boost::asio::io_context clangStderrStream;
    std::future<std::string> clangStderrFuture;

    bp::child clang(clangCmd, bp::std_in<stdinPipe, bp::std_out> bp::null,
                    bp::std_err > clangStderrFuture, clangStderrStream);

    // Write the IR to stdin.
    boost::asio::async_write(
        stdinPipe, stdinBuffer,
        [&](const boost::system::error_code& ec, std::size_t n) { stdinPipe.async_close(); });

    clangContext.run_for(std::chrono::seconds(60));
    if (clangContext.poll()) {
      return Status(StatusCode::INVALID_ARGUMENT,
                    "Failed to compute .text size cost within 60 seconds");
    }
    clang.wait();
    clangStderrStream.run();

    if (clang.exit_code()) {
      const std::string stderr = clangStderrFuture.get();
      return Status(StatusCode::INVALID_ARGUMENT,
                    fmt::format("Failed to compute .text size cost. "
                                "Command returned exit code {}: {}. Error: {}",
                                clang.exit_code(), clangCmd, stderr));
    }
  } catch (bp::process_error& e) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("Failed to compute .text size cost: {}", e.what()));
  }

  Status status = getTextSizeInBytes(tmpFile, value);
  fs::remove(tmpFile);
  return status;
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
      int64_t size;
      RETURN_IF_ERROR(getTextSizeInBytes(module, &size, workingDirectory, dynamicConfig,
                                         /*compile_only=*/true));
      *cost = static_cast<double>(size);
      break;
    }
    case LlvmCostFunction::TEXT_SIZE_BYTES: {
      int64_t size;
      RETURN_IF_ERROR(getTextSizeInBytes(module, &size, workingDirectory, dynamicConfig,
                                         /*compile_only=*/false));
      *cost = static_cast<double>(size);
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

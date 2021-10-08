// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/Benchmark.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <chrono>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/Subprocess.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/SHA1.h"

namespace fs = boost::filesystem;
namespace sys = boost::system;

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym::llvm_service {

namespace {

BenchmarkHash getModuleHash(const llvm::Module& module) {
  BenchmarkHash hash;
  llvm::SmallVector<char, 256> buffer;
  // Writing the entire bitcode to a buffer that is then discarded is
  // inefficient.
  llvm::BitcodeWriter writer(buffer);
  writer.writeModule(module, /*ShouldPreserveUseListOrder=*/false,
                     /*Index=*/nullptr, /*GenerateHash=*/true, &hash);
  return hash;
}

std::unique_ptr<llvm::Module> makeModuleOrDie(llvm::LLVMContext& context, const Bitcode& bitcode,
                                              const std::string& name) {
  Status status;
  auto module = makeModule(context, bitcode, name, &status);
  CHECK(status.ok()) << "Failed to make LLVM module: " << status.error_message();
  return module;
}

RealizedBenchmarkDynamicConfig realizeDynamicConfig(const BenchmarkDynamicConfig& original,
                                                    const fs::path& scratchDirectory) {
  BenchmarkDynamicConfig cfg;
  cfg.CopyFrom(original);

  // Set up the environment variables.
  (*cfg.mutable_build_cmd()->mutable_env())["CC"] =
      util::getSiteDataPath("llvm-v0/bin/clang").string();
  (*cfg.mutable_build_cmd()->mutable_env())["IN"] = (scratchDirectory / "out.bc").string();

  // Register the IR as a pre-requisite build file.
  cfg.mutable_build_cmd()->add_infile((scratchDirectory / "out.bc").string());

  return RealizedBenchmarkDynamicConfig(cfg);
}

}  // anonymous namespace

Status readBitcodeFile(const fs::path& path, Bitcode* bitcode) {
  std::ifstream ifs(path.string());
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("File not found: \"{}\"", path.string()));
  }

  ifs.seekg(0, std::ios::end);
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Error reading file: \"{}\"", path.string()));
  }

  std::streampos fileSize = ifs.tellg();
  if (!fileSize) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("File is empty: \"{}\"", path.string()));
  }

  bitcode->resize(fileSize);
  ifs.seekg(0);
  ifs.read(&(*bitcode)[0], bitcode->size());
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Error reading file: \"{}\"", path.string()));
  }

  return Status::OK;
}

std::unique_ptr<llvm::Module> makeModule(llvm::LLVMContext& context, const Bitcode& bitcode,
                                         const std::string& name, Status* status) {
  llvm::MemoryBufferRef buffer(llvm::StringRef(bitcode.data(), bitcode.size()), name);
  VLOG(3) << "llvm::parseBitcodeFile(" << bitcode.size() << " bits)";
  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOrError =
      llvm::parseBitcodeFile(buffer, context);
  if (moduleOrError) {
    *status = Status::OK;
    std::unique_ptr<llvm::Module> module = std::move(moduleOrError.get());

    // Strip the module identifiers and source file names from the module to
    // anonymize them. This is to deter learning algorithms from overfitting to
    // benchmarks by their name.
    module->setModuleIdentifier("-");
    module->setSourceFileName("-");

    // Strip module debug info.
    llvm::StripDebugInfo(*module);

    // Erase module-level named metadata.
    while (!module->named_metadata_empty()) {
      llvm::NamedMDNode* nmd = &*module->named_metadata_begin();
      module->eraseNamedMetadata(nmd);
    }

    return module;
  } else {
    *status = Status(StatusCode::INVALID_ARGUMENT,
                     fmt::format("Failed to parse LLVM bitcode: \"{}\"", name));
    return nullptr;
  }
}

// A benchmark is an LLVM module and the LLVM context that owns it.
Benchmark::Benchmark(const std::string& name, const Bitcode& bitcode,
                     const BenchmarkDynamicConfig& dynamicConfig, const fs::path& workingDirectory,
                     const BaselineCosts& baselineCosts)
    : context_(std::make_unique<llvm::LLVMContext>()),
      module_(makeModuleOrDie(*context_, bitcode, name)),
      scratchDirectory_(fs::path(fs::unique_path(workingDirectory / "scratch-%%%%-%%%%"))),
      dynamicConfigProto_(dynamicConfig),
      dynamicConfig_(realizeDynamicConfig(dynamicConfig, scratchDirectory_)),
      baselineCosts_(baselineCosts),
      name_(name),
      needsRecompile_(true),
      runtimesPerObservationCount_(kDefaultRuntimesPerObservationCount),
      warmupRunsPerRuntimeObservationCount_(kDefaultWarmupRunsPerRuntimeObservationCount),
      buildtimesPerObservationCount_(kDefaultBuildtimesPerObservationCount) {
  sys::error_code ec;
  fs::create_directory(scratchDirectory(), ec);
  CHECK(!ec) << "Failed to create scratch directory: " << scratchDirectory();
}

Benchmark::Benchmark(const std::string& name, std::unique_ptr<llvm::LLVMContext> context,
                     std::unique_ptr<llvm::Module> module,
                     const BenchmarkDynamicConfig& dynamicConfig, const fs::path& workingDirectory,
                     const BaselineCosts& baselineCosts)
    : context_(std::move(context)),
      module_(std::move(module)),
      scratchDirectory_(fs::path(fs::unique_path(workingDirectory / "scratch-%%%%-%%%%"))),
      dynamicConfigProto_(dynamicConfig),
      dynamicConfig_(realizeDynamicConfig(dynamicConfig, scratchDirectory_)),
      baselineCosts_(baselineCosts),
      name_(name),
      needsRecompile_(true) {
  sys::error_code ec;
  fs::create_directory(scratchDirectory(), ec);
  CHECK(!ec) << "Failed to create scratch directory: " << scratchDirectory();
}

std::unique_ptr<Benchmark> Benchmark::clone(const fs::path& workingDirectory) const {
  Bitcode bitcode;
  llvm::raw_svector_ostream ostream(bitcode);
  llvm::WriteBitcodeToFile(module(), ostream);

  return std::make_unique<Benchmark>(name(), bitcode, dynamicConfigProto_, workingDirectory,
                                     baselineCosts());
}

BenchmarkHash Benchmark::module_hash() const { return getModuleHash(*module_); }

Status Benchmark::verify_module() {
  std::string errorMessage;
  llvm::raw_string_ostream rso(errorMessage);
  if (llvm::verifyModule(module(), &rso)) {
    rso.flush();
    return Status(StatusCode::DATA_LOSS, "Failed to verify module: " + errorMessage);
  }
  return Status::OK;
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

Status Benchmark::writeBitcodeToFile(const fs::path& path) {
  return writeBitcodeFile(module(), path);
}

Status Benchmark::computeRuntime(Observation& observation) {
  const RealizedBenchmarkDynamicConfig& cfg = dynamicConfig();

  if (!cfg.isRunnable()) {
    return Status::OK;
  }

  if (chdir(scratchDirectory().string().c_str())) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to set working directory: {}", scratchDirectory().string()));
  }

  RETURN_IF_ERROR(compile());

  // Run the pre-execution hooks.
  for (const auto& preRunCommand : cfg.preRunCommands()) {
    RETURN_IF_ERROR(preRunCommand.checkInfiles());
    RETURN_IF_ERROR(preRunCommand.checkCall());
    RETURN_IF_ERROR(preRunCommand.checkOutfiles());
  }

  RETURN_IF_ERROR(cfg.runCommand().checkInfiles());

  // Run the warmup runs.
  VLOG(3) << "Running " << getWarmupRunsPerRuntimeObservationCount()
          << " warmup iterations of binary";
  for (int i = 0; i < getRuntimesPerObservationCount(); ++i) {
    RETURN_IF_ERROR(cfg.runCommand().checkCall());
  }

  // Run the binary.
  VLOG(3) << "Running " << getRuntimesPerObservationCount() << " iterations of binary";
  for (int i = 0; i < getRuntimesPerObservationCount(); ++i) {
    const auto startTime = std::chrono::steady_clock::now();
    RETURN_IF_ERROR(cfg.runCommand().checkCall());
    const auto endTime = std::chrono::steady_clock::now();
    const auto elapsedMicroseconds =
        std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
    observation.mutable_double_list()->add_value(static_cast<double>(elapsedMicroseconds) /
                                                 1000000);
  }

  RETURN_IF_ERROR(cfg.runCommand().checkOutfiles());

  // Run the post-execution hooks.
  for (const auto& postRunCommand : cfg.postRunCommands()) {
    RETURN_IF_ERROR(postRunCommand.checkInfiles());
    RETURN_IF_ERROR(postRunCommand.checkCall());
    RETURN_IF_ERROR(postRunCommand.checkOutfiles());
  }

  return Status::OK;
}

Status Benchmark::computeBuildtime(Observation& observation) {
  if (!dynamicConfig().isBuildable()) {
    return Status::OK;
  }

  RETURN_IF_ERROR(compile());

  observation.mutable_double_list()->add_value(static_cast<double>(lastBuildTimeMicroseconds()) /
                                               1000000);

  return Status::OK;
}

Status Benchmark::compile() {
  const auto& cfg = dynamicConfig();

  if (!cfg.isBuildable()) {
    return Status::OK;
  }

  if (!needsRecompile_) {
    return Status::OK;
  }

  VLOG(3) << "Compiling benchmark";

  if (chdir(scratchDirectory().string().c_str())) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to set working directory: {}", scratchDirectory().string()));
  }

  // Write the bitcode to a file.
  RETURN_IF_ERROR(writeBitcodeToFile(scratchDirectory() / "out.bc"));

  // Check that the required sources exist.
  RETURN_IF_ERROR(cfg.buildCommand().checkInfiles());

  // Build the bitcode.
  const std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(cfg.buildCommand().checkCall());
  const auto end = std::chrono::steady_clock::now();

  // Check that the expected output files were generated.
  RETURN_IF_ERROR(cfg.buildCommand().checkOutfiles());

  buildTimeMicroseconds_ =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  needsRecompile_ = false;
  return Status::OK;
}

bool Benchmark::applyBaselineOptimizations(unsigned optLevel, unsigned sizeLevel) {
  return applyBaselineOptimizationsToModule(&module(), optLevel, sizeLevel);
}

namespace {

std::vector<util::LocalShellCommand> commandsFromProto(
    const google::protobuf::RepeatedPtrField<Command>& cmds) {
  std::vector<util::LocalShellCommand> outs;
  for (const auto& cmd : cmds) {
    outs.push_back(util::LocalShellCommand(cmd));
  }
  return outs;
}

}  // anonymous namespace

RealizedBenchmarkDynamicConfig::RealizedBenchmarkDynamicConfig(const BenchmarkDynamicConfig& cfg)
    : buildCommand_(cfg.build_cmd()),
      runCommand_(cfg.run_cmd()),
      preRunCommands_(commandsFromProto(cfg.pre_run_cmd())),
      postRunCommands_(commandsFromProto(cfg.post_run_cmd())),
      isBuildable_(!buildCommand_.empty()),
      isRunnable_(!(buildCommand_.empty() || runCommand_.empty())) {}

}  // namespace compiler_gym::llvm_service

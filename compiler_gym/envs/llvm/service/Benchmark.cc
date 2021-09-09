// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/Benchmark.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <stdexcept>
#include <subprocess/subprocess.hpp>
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

void substringReplace(const std::string& pattern, const std::string& replacement,
                      std::string& str) {
  size_t pos = 0;
  while (true) {
    pos = str.find(pattern, pos);
    if (pos == std::string::npos) {
      break;
    }

    str.replace(pos, pattern.size(), replacement);
    pos += pattern.size();
  }
}

std::string expandBuildCommand(const std::string& cmd, const fs::path& scratchDirectory) {
  std::string out = cmd;
  const auto clangPath = util::getSiteDataPath("llvm-v0/bin/clang");
  substringReplace("$CC", clangPath.string(), out);
  substringReplace("$<", (scratchDirectory / "out.bc").string(), out);
  return out;
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
      dynamicConfig_(dynamicConfig),
      scratchDirectory_(fs::unique_path(workingDirectory / "scratch-%%%%-%%%%")),
      buildCmd_(expandBuildCommand(dynamicConfig.build_cmd(), scratchDirectory_)),
      isBuildable_(dynamicConfig.build_cmd().size()),
      isRunnable_(dynamicConfig.build_cmd().size() && dynamicConfig.run_cmd().size()),
      baselineCosts_(baselineCosts),
      name_(name),
      needsRecompile_(true),
      runtimesPerObservationCount_(kDefaultRuntimesPerObservationCount),
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
      dynamicConfig_(dynamicConfig),
      scratchDirectory_(fs::unique_path(workingDirectory / "scratch-%%%%-%%%%")),
      buildCmd_(expandBuildCommand(dynamicConfig.build_cmd(), scratchDirectory_)),
      isBuildable_(dynamicConfig.build_cmd().size()),
      isRunnable_(dynamicConfig.build_cmd().size() && dynamicConfig.run_cmd().size()),
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

  return std::make_unique<Benchmark>(name(), bitcode, dynamicConfig(), workingDirectory,
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
  if (!isRunnable()) {
    return Status::OK;
  }

  RETURN_IF_ERROR(compile());

  // Run the pre-execution hooks.
  for (int i = 0; i < dynamicConfig().pre_run_cmd_size(); ++i) {
    RETURN_IF_ERROR(util::checkCall(dynamicConfig().pre_run_cmd(i),
                                    dynamicConfig().pre_run_cmd_timeout_seconds(),
                                    scratchDirectory()));
  }

  // Run the binary.
  VLOG(3) << "Run: " << dynamicConfig().run_cmd();
  for (int i = 0; i < getRuntimesPerObservationCount(); ++i) {
    const std::chrono::time_point<std::chrono::steady_clock> start =
        std::chrono::steady_clock::now();
    RETURN_IF_ERROR(util::checkCall(dynamicConfig().run_cmd(),
                                    dynamicConfig().run_cmd_timeout_seconds(), scratchDirectory()));
    const auto end = std::chrono::steady_clock::now();
    observation.mutable_double_list()->add_value(
        static_cast<double>(
            std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()) /
        1000000);
  }

  // Run the post-execution hooks.
  for (int i = 0; i < dynamicConfig().post_run_cmd_size(); ++i) {
    RETURN_IF_ERROR(util::checkCall(dynamicConfig().post_run_cmd(i),
                                    dynamicConfig().post_run_cmd_timeout_seconds(),
                                    scratchDirectory()));
  }

  return Status::OK;
}

Status Benchmark::computeBuildtime(Observation& observation) {
  if (!isBuildable()) {
    return Status::OK;
  }

  RETURN_IF_ERROR(compile());

  observation.mutable_double_list()->add_value(static_cast<double>(lastBuildTimeMicroseconds()) /
                                               1000000);

  return Status::OK;
}

Status Benchmark::compile() {
  if (!isRunnable()) {
    return Status::OK;
  }

  if (!needsRecompile_) {
    return Status::OK;
  }

  // Write the bitcode to a file.
  RETURN_IF_ERROR(writeBitcodeToFile(scratchDirectory() / "out.bc"));
  VLOG(3) << "Compile: " << buildCmd_;

  // Build the bitcode.
  const std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(
      util::checkCall(buildCmd_, dynamicConfig().build_cmd_timeout_seconds(), scratchDirectory()));
  const auto end = std::chrono::steady_clock::now();

  // If an output file path was specified, check that is generated.
  if (dynamicConfig().build_genfile().size()) {
    fs::path bin{dynamicConfig().build_genfile()};
    if (!bin.is_absolute()) {
      bin = scratchDirectory() / bin;
    }
    if (!fs::exists(bin)) {
      return Status(StatusCode::INTERNAL,
                    fmt::format("Compilation command '{}' did not produce expected file: {}",
                                buildCmd_, bin.string()));
    }
  }

  buildTimeMicroseconds_ =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  needsRecompile_ = false;
  return Status::OK;
}

bool Benchmark::applyBaselineOptimizations(unsigned optLevel, unsigned sizeLevel) {
  return applyBaselineOptimizationsToModule(&module(), optLevel, sizeLevel);
}

}  // namespace compiler_gym::llvm_service

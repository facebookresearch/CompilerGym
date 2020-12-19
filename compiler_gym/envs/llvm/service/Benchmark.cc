// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/Benchmark.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <stdexcept>

#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/SHA1.h"

namespace fs = boost::filesystem;
using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym::llvm_service {

namespace {

BaselineCosts getBaselineCosts(const llvm::Module& unoptimizedModule,
                               const fs::path& workingDirectory) {
  BaselineCosts baselineCosts;
  setbaselineCosts(unoptimizedModule, &baselineCosts, workingDirectory);
  return baselineCosts;
}

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
  return std::move(module);
}

}  // anonymous namespace

std::unique_ptr<llvm::Module> makeModule(llvm::LLVMContext& context, const Bitcode& bitcode,
                                         const std::string& name, Status* status) {
  llvm::MemoryBufferRef buffer(llvm::StringRef(bitcode.data(), bitcode.size()), name);
  VLOG(3) << "llvm::parseBitcodeFile(" << bitcode.size() << " bits)";
  llvm::Expected<std::unique_ptr<llvm::Module>> moduleOrError =
      llvm::parseBitcodeFile(buffer, context);
  if (moduleOrError) {
    *status = Status::OK;
    return std::move(moduleOrError.get());
  } else {
    *status = Status(StatusCode::INVALID_ARGUMENT,
                     fmt::format("Failed to parse LLVM bitcode: \"{}\"", name));
    return nullptr;
  }
}

// A benchmark is an LLVM module and the LLVM context that owns it.
Benchmark::Benchmark(const std::string& name, const Bitcode& bitcode,
                     const fs::path& workingDirectory, std::optional<fs::path> bitcodePath,
                     const BaselineCosts* baselineCosts)
    : context_(std::make_unique<llvm::LLVMContext>()),
      module_(makeModuleOrDie(*context_, bitcode, name)),
      baselineCosts_(baselineCosts ? *baselineCosts : getBaselineCosts(*module_, workingDirectory)),
      hash_(getModuleHash(*module_)),
      name_(name),
      bitcodeSize_(bitcode.size()),
      bitcodePath_(bitcodePath) {}

Benchmark::Benchmark(const std::string& name, std::unique_ptr<llvm::LLVMContext> context,
                     std::unique_ptr<llvm::Module> module, size_t bitcodeSize,
                     const fs::path& workingDirectory, std::optional<fs::path> bitcodePath,
                     const BaselineCosts* baselineCosts)
    : context_(std::move(context)),
      module_(std::move(module)),
      baselineCosts_(baselineCosts ? *baselineCosts : getBaselineCosts(*module_, workingDirectory)),
      hash_(getModuleHash(*module_)),
      name_(name),
      bitcodeSize_(bitcodeSize),
      bitcodePath_(bitcodePath) {}

std::unique_ptr<Benchmark> Benchmark::clone(const fs::path& workingDirectory) const {
  Bitcode bitcode;
  llvm::raw_svector_ostream ostream(bitcode);
  llvm::WriteBitcodeToFile(module(), ostream);

  return std::make_unique<Benchmark>(name(), bitcode, workingDirectory, bitcodePath(),
                                     &baselineCosts());
}

}  // namespace compiler_gym::llvm_service

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <optional>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "include/llvm/IR/ModuleSummaryIndex.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace compiler_gym::llvm_service {

// We identify benchmarks using a hash of the LLVM module, which is a
// 160 bits SHA1.
//
// NOTE(cummins): In the future when we extend this to support optimizing for
// performance, we would need this
using BenchmarkHash = llvm::ModuleHash;

using Bitcode = llvm::SmallString<0>;

grpc::Status readBitcodeFile(const boost::filesystem::path& path, Bitcode* bitcode);

// Parses the given bitcode into a module and strips the identifying ModuleID
// and source_filename attributes. Returns nullptr on error and sets status.
std::unique_ptr<llvm::Module> makeModule(llvm::LLVMContext& context, const Bitcode& bitcode,
                                         const std::string& name, grpc::Status* status);

// A benchmark is an LLVM module and the LLVM context that owns it. A benchmark
// is mutable and can be changed over the course of a session.
class Benchmark {
 public:
  Benchmark(const std::string& name, const Bitcode& bitcode,
            const boost::filesystem::path& workingDirectory, const BaselineCosts& baselineCosts);

  Benchmark(const std::string& name, std::unique_ptr<llvm::LLVMContext> context,
            std::unique_ptr<llvm::Module> module, size_t bitcodeSize,
            const boost::filesystem::path& workingDirectory, const BaselineCosts& baselineCosts);

  // Make a copy of the benchmark.
  std::unique_ptr<Benchmark> clone(const boost::filesystem::path& workingDirectory) const;

  inline const std::string& name() const { return name_; }

  inline const size_t bitcodeSize() const { return bitcodeSize_; }

  inline llvm::Module& module() { return *module_; }

  inline const llvm::Module& module() const { return *module_; }

  inline llvm::LLVMContext& context() { return *context_; }

  inline const llvm::LLVMContext& context() const { return *context_; }

  inline const BaselineCosts& baselineCosts() const { return baselineCosts_; }

  // Accessors for the underlying raw pointers.
  inline const llvm::LLVMContext* context_ptr() const { return context_.get(); }

  inline const llvm::Module* module_ptr() const { return module_.get(); }

  inline const BenchmarkHash hash() const { return hash_; }

  // Replace the benchmark module with a new one. This is to enable
  // out-of-process modification of the IR by serializing the benchmark to a
  // file, modifying the file, then loading the modified file and updating the
  // module pointer here.
  inline void replaceModule(std::unique_ptr<llvm::Module> module) { module_ = std::move(module); }

 private:
  // NOTE(cummins): Order here is important! The LLVMContext must be declared
  // before Module, as class members are destroyed in the reverse order they are
  // declared, and a module must never outlive its context.
  std::unique_ptr<llvm::LLVMContext> context_;
  std::unique_ptr<llvm::Module> module_;
  const BaselineCosts baselineCosts_;
  const BenchmarkHash hash_;
  const std::string name_;
  // The length of the bitcode string for this benchmark.
  const size_t bitcodeSize_;
};

}  // namespace compiler_gym::llvm_service

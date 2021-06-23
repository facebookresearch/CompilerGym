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

/**
 * A 160 bits SHA1 that identifies an LLVM module.
 */
using BenchmarkHash = llvm::ModuleHash;

/**
 * A bitcode.
 */
using Bitcode = llvm::SmallString<0>;

/**
 * Read a bitcode file from disk.
 *
 * @param path The path of the bitcode file to read.
 * @param bitcode The destination bitcode.
 * @return `OK` on success, `NOT_FOUND` if the file is not found, or
 *     `INVALID_ARGUMENT` if the file is invalid.
 */
grpc::Status readBitcodeFile(const boost::filesystem::path& path, Bitcode* bitcode);

/**
 * Construct an LLVM module from a bitcode.
 *
 * Parses the given bitcode into a module and strips the identifying `ModuleID`
 * and `source_filename` attributes.
 *
 * @param context An LLVM context for the new module.
 * @param bitcode The bitcode to parse.
 * @param name The name of the module.
 * @param status An error status that is set to `OK` on success or
 *    `INVALID_ARGUMENT` if the bitcode cannot be parsed.
 * @return A unique pointer to an LLVM module, or `nullptr` on error and sets
 *    `status`.
 */
std::unique_ptr<llvm::Module> makeModule(llvm::LLVMContext& context, const Bitcode& bitcode,
                                         const std::string& name, grpc::Status* status);

/**
 * An LLVM module and the LLVM context that owns it.
 *
 * A benchmark is mutable and can be changed over the course of a session.
 */
class Benchmark {
 public:
  /**
   * Construct a benchmark from a bitcode.
   */
  Benchmark(const std::string& name, const Bitcode& bitcode,
            const boost::filesystem::path& workingDirectory, const BaselineCosts& baselineCosts);

  /**
   * Construct a benchmark from an LLVM module.
   */
  Benchmark(const std::string& name, std::unique_ptr<llvm::LLVMContext> context,
            std::unique_ptr<llvm::Module> module, const boost::filesystem::path& workingDirectory,
            const BaselineCosts& baselineCosts);

  /**
   * Make a copy of the benchmark.
   *
   * @param workingDirectory The working directory for the new benchmark.
   * @return A copy of the benchmark.
   */
  std::unique_ptr<Benchmark> clone(const boost::filesystem::path& workingDirectory) const;

  /**
   * Compute and return a SHA1 hash of the module.
   *
   * @return A SHA1 hash of the module.
   */
  BenchmarkHash module_hash() const;

  /**
   * Wrapper around `llvm::verifyModule()` which returns an error status on
   * failure.
   *
   * @return `OK` on success, else `DATA_LOSS` if verification fails.
   */
  grpc::Status verify_module();

  /**
   * The name of the benchmark.
   */
  inline const std::string& name() const { return name_; }

  /**
   * The underlying LLVM module.
   */
  inline llvm::Module& module() { return *module_; }

  /**
   * The underlying LLVM module.
   */
  inline const llvm::Module& module() const { return *module_; }

  /**
   * The underlying LLVM context.
   */
  inline llvm::LLVMContext& context() { return *context_; }

  /**
   * The underlying LLVM context.
   */
  inline const llvm::LLVMContext& context() const { return *context_; }

  inline const BaselineCosts& baselineCosts() const { return baselineCosts_; }

  // Accessors for the underlying raw pointers.

  /**
   * A pointer to the underlying LLVM context.
   */
  inline const llvm::LLVMContext* context_ptr() const { return context_.get(); }

  /**
   * A pointer to the underlying LLVM module.
   */
  inline const llvm::Module* module_ptr() const { return module_.get(); }

  /** Replace the benchmark module with a new one.
   *
   * This is to enable out-of-process modification of the IR by serializing the
   * benchmark to a file, modifying the file, then loading the modified file and
   * updating the module pointer here.
   *
   * @param module A new module.
   */
  inline void replaceModule(std::unique_ptr<llvm::Module> module) { module_ = std::move(module); }

 private:
  // NOTE(cummins): Order here is important! The LLVMContext must be declared
  // before Module, as class members are destroyed in the reverse order they are
  // declared, and a module must never outlive its context.
  std::unique_ptr<llvm::LLVMContext> context_;
  std::unique_ptr<llvm::Module> module_;
  const BaselineCosts baselineCosts_;
  const std::string name_;
};

}  // namespace compiler_gym::llvm_service

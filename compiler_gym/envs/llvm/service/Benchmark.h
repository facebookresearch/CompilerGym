// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <optional>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/BenchmarkDynamicConfig.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/Subprocess.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"

namespace compiler_gym::llvm_service {

/**
 * A 160 bits SHA1 that identifies an LLVM module.
 */
using BenchmarkHash = llvm::ModuleHash;

/**
 * A bitcode.
 */
using Bitcode = llvm::SmallString<0>;

/** The number of times a benchmark is executed. This can be overriden using
 * the "llvm.set_runtimes_per_observation_count" session parameter.
 */
constexpr int kDefaultRuntimesPerObservationCount = 1;

/** The default number of warmup runs that a benchmark is executed before measuring the runtimes.
 * This can be overriden using the "llvm.set_warmup_runs_count_per_runtime_observation" session
 * parameter.
 */
constexpr int kDefaultWarmupRunsPerRuntimeObservationCount = 0;

/** The number of times a benchmark is built. This can be overriden using
 * the "llvm.set_buildtimes_per_observation_count" session parameter.
 */
constexpr int kDefaultBuildtimesPerObservationCount = 1;

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
 * Write the module bitcode to the given path.
 *
 * @param module The module to write to file.
 * @param path The path of the bitcode file to write.
 * @return `OK` on success.
 */
grpc::Status writeBitcodeFile(const llvm::Module& module, const boost::filesystem::path& path);

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
            const compiler_gym::BenchmarkDynamicConfig& dynamicConfig,
            const boost::filesystem::path& workingDirectory, const BaselineCosts& baselineCosts);

  /**
   * Construct a benchmark from an LLVM module.
   */
  Benchmark(const std::string& name, std::unique_ptr<llvm::LLVMContext> context,
            std::unique_ptr<llvm::Module> module,
            const compiler_gym::BenchmarkDynamicConfig& dynamicConfig,
            const boost::filesystem::path& workingDirectory, const BaselineCosts& baselineCosts);

  void close();

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
   * Write the module bitcode to the given path.
   */
  grpc::Status writeBitcodeToFile(const boost::filesystem::path& path);

  /**
   * Compute a list of runtimes.
   *
   * If the benchmark is not runnable, the list is empty.
   */
  grpc::Status computeRuntime(Event& observation);

  /**
   * Compute a list of buildtimes.
   *
   * If the benchmark is not buildable, the list is empty.
   */
  grpc::Status computeBuildtime(Event& observation);

  grpc::Status compile();

  /**
   * Apply the given baseline optimizations.
   *
   * @param optLevel The runtime optimization level.
   * @param sizeLevel The size optimization level
   * @return Whether the baseline optimizations modified the module.
   */
  bool applyBaselineOptimizations(unsigned optLevel, unsigned sizeLevel);

  /**
   * The name of the benchmark.
   */
  inline const std::string& name() const { return name_; }

  /**
   * Mark that the LLVM module has been modified.
   */
  inline void markModuleModified() { needsRecompile_ = true; }

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

  /**
   * A reference to the dynamic configuration object.
   */
  inline const BenchmarkDynamicConfig& dynamicConfig() const { return dynamicConfig_; }

  inline bool isBuildable() const { return dynamicConfig().isBuildable(); }

  inline bool isRunnable() const { return dynamicConfig().isRunnable(); }

  /** Replace the benchmark module with a new one.
   *
   * This is to enable out-of-process modification of the IR by serializing the
   * benchmark to a file, modifying the file, then loading the modified file and
   * updating the module pointer here.
   *
   * @param module A new module.
   */
  inline void replaceModule(std::unique_ptr<llvm::Module> module) {
    module_ = std::move(module);
    markModuleModified();
  }

  inline int64_t lastBuildTimeMicroseconds() { return buildTimeMicroseconds_; }

  inline int getRuntimesPerObservationCount() const { return runtimesPerObservationCount_; }

  inline void setRuntimesPerObservationCount(const int value) {
    runtimesPerObservationCount_ = value;
  }

  inline int getWarmupRunsPerRuntimeObservationCount() const {
    return warmupRunsPerRuntimeObservationCount_;
  }

  inline void setWarmupRunsPerRuntimeObservationCount(const int value) {
    warmupRunsPerRuntimeObservationCount_ = value;
  }

  inline int getBuildtimesPerObservationCount() const { return buildtimesPerObservationCount_; }

  inline void setBuildtimesPerObservationCount(const int value) {
    buildtimesPerObservationCount_ = value;
  }

 private:
  inline const boost::filesystem::path& scratchDirectory() const { return scratchDirectory_; }
  inline const boost::filesystem::path workingDirectory() const {
    return scratchDirectory_.parent_path();
  }

  // NOTE(cummins): Order here is important! The LLVMContext must be declared
  // before Module, as class members are destroyed in the reverse order they are
  // declared, and a module must never outlive its context.
  std::unique_ptr<llvm::LLVMContext> context_;
  std::unique_ptr<llvm::Module> module_;
  const boost::filesystem::path scratchDirectory_;
  const compiler_gym::BenchmarkDynamicConfig dynamicConfigProto_;
  const BenchmarkDynamicConfig dynamicConfig_;
  const BaselineCosts baselineCosts_;
  /** The directory used for storing build / runtime artifacts. The difference
   * between the scratch directory and the working directory is that the working
   * directory may be shared across multiple Benchmark instances. The scratch
   * directory is unique.
   */
  const std::string name_;
  bool needsRecompile_;
  int64_t buildTimeMicroseconds_;
  int runtimesPerObservationCount_;
  int warmupRunsPerRuntimeObservationCount_;
  int buildtimesPerObservationCount_;
};

}  // namespace compiler_gym::llvm_service

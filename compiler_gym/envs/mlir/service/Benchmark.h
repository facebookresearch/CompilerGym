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
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/Subprocess.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/ModuleSummaryIndex.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

namespace compiler_gym::mlir_service {

/**
 * A bitcode.
 */
using Bitcode = std::string;

/** The number of times a benchmark is executed. This can be overriden using
 * the "mlir.set_runtimes_per_observation_count" session parameter.
 */
constexpr int kDefaultRuntimesPerObservationCount = 1;

/** The default number of warmup runs that a benchmark is executed before measuring the runtimes.
 * This can be overriden using the "mlir.set_warmup_runs_count_per_runtime_observation" session
 * parameter.
 */
constexpr int kDefaultWarmupRunsPerRuntimeObservationCount = 0;

/** The number of times a benchmark is built. This can be overriden using
 * the "mlir.set_buildtimes_per_observation_count" session parameter.
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
grpc::Status writeBitcodeFile(mlir::OwningModuleRef& module, const boost::filesystem::path& path);

/**
 * Construct an MLIR module from a bitcode.
 *
 * Parses the given bitcode into a module and strips the identifying `ModuleID`
 * and `source_filename` attributes.
 *
 * @param context An MLIR context for the new module.
 * @param bitcode The bitcode to parse.
 * @param name The name of the module.
 * @param status An error status that is set to `OK` on success or
 *    `INVALID_ARGUMENT` if the bitcode cannot be parsed.
 * @return A unique pointer to an MLIR module, or `nullptr` on error and sets
 *    `status`.
 */
std::unique_ptr<mlir::OwningModuleRef> makeModule(mlir::MLIRContext& context,
                                                  const Bitcode& bitcode, const std::string& name,
                                                  grpc::Status* status);

/**
 * Represents a BenchmarkDynamicConfig protocol buffer.
 */
class RealizedBenchmarkDynamicConfig {
 public:
  explicit RealizedBenchmarkDynamicConfig(const BenchmarkDynamicConfig& cfg);

  inline const util::LocalShellCommand& buildCommand() const { return buildCommand_; };
  inline const util::LocalShellCommand& runCommand() const { return runCommand_; };
  inline const std::vector<util::LocalShellCommand>& preRunCommands() const {
    return preRunCommands_;
  };
  inline const std::vector<util::LocalShellCommand>& postRunCommands() const {
    return postRunCommands_;
  };
  inline bool isBuildable() const { return isBuildable_; }
  inline bool isRunnable() const { return isRunnable_; }

 private:
  const util::LocalShellCommand buildCommand_;
  const util::LocalShellCommand runCommand_;
  const std::vector<util::LocalShellCommand> preRunCommands_;
  const std::vector<util::LocalShellCommand> postRunCommands_;
  const bool isBuildable_;
  const bool isRunnable_;
};

/**
 * An MLIR module and the MLIR context that owns it.
 *
 * A benchmark is mutable and can be changed over the course of a session.
 */
class Benchmark {
 public:
  /**
   * Construct a benchmark from a bitcode.
   */
  Benchmark(const std::string& name, const Bitcode& bitcode,
            const BenchmarkDynamicConfig& dynamicConfig,
            const boost::filesystem::path& workingDirectory);

  /**
   * Construct a benchmark from an MLIR module.
   */
  Benchmark(const std::string& name, std::unique_ptr<mlir::MLIRContext> context,
            std::unique_ptr<mlir::OwningModuleRef> module,
            const BenchmarkDynamicConfig& dynamicConfig,
            const boost::filesystem::path& workingDirectory);

  /**
   * Make a copy of the benchmark.
   *
   * @param workingDirectory The working directory for the new benchmark.
   * @return A copy of the benchmark.
   */
  std::unique_ptr<Benchmark> clone(const boost::filesystem::path& workingDirectory);

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
   * Mark that the MLIR module has been modified.
   */
  inline void markModuleModified() { needsRecompile_ = true; }

  /**
   * The underlying MLIR module.
   */
  inline mlir::OwningModuleRef& module() { return *module_; }

  /**
   * The underlying MLIR module.
   */
  inline const mlir::OwningModuleRef& module() const { return *module_; }

  /**
   * The underlying MLIR context.
   */
  inline mlir::MLIRContext& context() { return *context_; }

  /**
   * The underlying MLIR context.
   */
  inline const mlir::MLIRContext& context() const { return *context_; }

  // Accessors for the underlying raw pointers.

  /**
   * A pointer to the underlying MLIR context.
   */
  inline const mlir::MLIRContext* context_ptr() const { return context_.get(); }

  /**
   * A pointer to the underlying MLIR module.
   */
  inline const mlir::OwningModuleRef* module_ptr() const { return module_.get(); }

  /**
   * A reference to the dynamic configuration object.
   */
  inline const RealizedBenchmarkDynamicConfig& dynamicConfig() const { return dynamicConfig_; }

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
  inline void replaceModule(std::unique_ptr<mlir::OwningModuleRef> module) {
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
  std::unique_ptr<mlir::MLIRContext> context_;
  std::unique_ptr<mlir::OwningModuleRef> module_;
  const boost::filesystem::path scratchDirectory_;
  const BenchmarkDynamicConfig dynamicConfigProto_;
  const RealizedBenchmarkDynamicConfig dynamicConfig_;
  /** The directory used for storing build / runtime artifacts. The difference
   * between the scratch directory and the working directory is that the working
   * directory may be shared across multiple Benchmark instances. The scratch
   * directory is unique.
   */
  const std::string name_;
  int m_, n_, k_;
  bool needsRecompile_;
  int64_t buildTimeMicroseconds_;
  int runtimesPerObservationCount_;
  int warmupRunsPerRuntimeObservationCount_;
  int buildtimesPerObservationCount_;
};

}  // namespace compiler_gym::mlir_service

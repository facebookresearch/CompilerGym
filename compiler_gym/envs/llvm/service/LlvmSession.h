// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <magic_enum.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/envs/llvm/service/Observation.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/service/CompilationSession.h"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"

namespace compiler_gym::llvm_service {

/**
 * An interactive LLVM compilation session.
 *
 * This class exposes the LLVM optimization pipeline for an LLVM module as an
 * interactive environment. It can be used directly as a C++ API, or it can be
 * accessed through an RPC interface using the CompilerGym RPC runtime.
 */
class LlvmSession final : public CompilationSession {
 public:
  LlvmSession(const boost::filesystem::path& workingDirectory);

  std::string getCompilerVersion() const final override;

  std::vector<ActionSpace> getActionSpaces() const final override;

  std::vector<ObservationSpace> getObservationSpaces() const final override;

  [[nodiscard]] grpc::Status init(const ActionSpace& actionSpace,
                                  const compiler_gym::Benchmark& benchmark) final override;

  [[nodiscard]] grpc::Status init(CompilationSession* other) final override;

  [[nodiscard]] grpc::Status applyAction(const Event& action, bool& endOfEpisode,
                                         std::optional<ActionSpace>& newActionSpace,
                                         bool& actionHadNoEffect) final override;

  [[nodiscard]] grpc::Status endOfStep(bool actionHadNoEffect, bool& endOfEpisode,
                                       std::optional<ActionSpace>& newActionSpace) final override;

  [[nodiscard]] grpc::Status computeObservation(const ObservationSpace& observationSpace,
                                                Event& observation) final override;

  [[nodiscard]] virtual grpc::Status handleSessionParameter(
      const std::string& key, const std::string& value,
      std::optional<std::string>& reply) final override;

  inline const LlvmActionSpace actionSpace() const { return actionSpace_; }

 private:
  [[nodiscard]] grpc::Status computeObservation(LlvmObservationSpace observationSpace,
                                                Event& observation);

  [[nodiscard]] grpc::Status init(const LlvmActionSpace& actionSpace,
                                  std::unique_ptr<Benchmark> benchmark);

  inline const Benchmark& benchmark() const {
    DCHECK(benchmark_) << "Calling benchmark() before init()";
    return *benchmark_;
  }
  inline Benchmark& benchmark() {
    DCHECK(benchmark_) << "Calling benchmark() before init()";
    return *benchmark_;
  }

  /**
   * Run the requested action.
   *
   * @param action An action to apply.
   * @param actionHadNoEffect Set to true if LLVM reported that any passes that
   *    were run made no modifications to the module.
   * @return `OK` on success.
   */
  [[nodiscard]] grpc::Status applyPassAction(LlvmAction action, bool& actionHadNoEffect);

  /**
   * Run the given pass, possibly modifying the underlying LLVM module.
   *
   * @return Whether the module was modified.
   */
  bool runPass(llvm::Pass* pass);

  /**
   * Run the given pass, possibly modifying the underlying LLVM module.
   *
   * @return Whether the module was modified.
   */
  bool runPass(llvm::FunctionPass* pass);

  /**
   * Run the commandline `opt` tool on the current LLVM module with the given
   * arguments, replacing the environment state with the generated output.
   */
  [[nodiscard]] grpc::Status runOptWithArgs(const std::vector<std::string>& optArgs);

  inline const llvm::TargetLibraryInfoImpl& tlii() const { return tlii_; }

  /**
   * Setup pass manager with depdendent passes and the specified pass.
   */
  template <typename PassManager, typename Pass>
  inline void setupPassManager(PassManager* passManager, Pass* pass) {
    passManager->add(new llvm::ProfileSummaryInfoWrapperPass());
    passManager->add(new llvm::TargetLibraryInfoWrapperPass(tlii()));
    passManager->add(createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));
    passManager->add(pass);
  }

  // Immutable state.
  const std::unordered_map<std::string, LlvmObservationSpace> observationSpaceNames_;
  // Mutable state initialized in init().
  LlvmActionSpace actionSpace_;
  std::unique_ptr<Benchmark> benchmark_;
  llvm::TargetLibraryInfoImpl tlii_;
};

}  // namespace compiler_gym::llvm_service

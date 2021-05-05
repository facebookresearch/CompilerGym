// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <magic_enum.hpp>
#include <memory>
#include <optional>

#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/service/core/Core.h"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "programl/proto/program_graph_options.pb.h"

namespace compiler_gym::llvm_service {

// This class exposes the LLVM optimization pipeline for an LLVM module as an
// interactive environment.
//
// It can be used directly as a C++ API, or it can be accessed through an RPC
// interface using the compiler_gym::service::LlvmService class.
class LlvmSession final : public CompilationSession {
 public:
  LlvmSession(const boost::filesystem::path& workingDirectory);

  std::string getCompilerVersion() const override;

  std::vector<ActionSpace> getActionSpaces() const override;

  ActionSpace getActionSpace() const override;

  std::vector<ObservationSpace> getObservationSpaces() const override;

  [[nodiscard]] grpc::Status init(size_t actionSpaceIndex,
                                  const compiler_gym::Benchmark& benchmark) override;

  [[nodiscard]] grpc::Status init(CompilationSession& other) override;

  [[nodiscard]] grpc::Status applyAction(size_t actionIndex, bool* endOfEpisode,
                                         bool* actionSpaceChanged,
                                         bool* actionHadNoEffect) override;

  [[nodiscard]] grpc::Status setObservation(size_t observationSpaceIndex,
                                            Observation* observation) override;

  inline const LlvmActionSpace actionSpace() const { return actionSpace_; }

  // Compute the requested observation.
  [[nodiscard]] grpc::Status getObservation(LlvmObservationSpace space, Observation* reply);

 private:
  [[nodiscard]] grpc::Status init(size_t actionSpaceIndex, std::unique_ptr<Benchmark> benchmark);

  inline const Benchmark& benchmark() const { return *benchmark_; }
  inline Benchmark& benchmark() { return *benchmark_; }

  // Run the requested action.
  [[nodiscard]] grpc::Status applyPassAction(LlvmAction action, bool* actionHadNoEffect);

  // Run the given pass, possibly modifying the underlying LLVM module.
  bool runPass(llvm::Pass* pass);
  bool runPass(llvm::FunctionPass* pass);

  // Run the commandline `opt` tool on the current LLVM module with the given
  // arguments, replacing the environment state with the generated output.
  [[nodiscard]] grpc::Status runOptWithArgs(const std::vector<std::string>& optArgs);

  inline const llvm::TargetLibraryInfoImpl& tlii() const { return tlii_; }

  // Setup pass manager with depdendent passes and the specified pass.
  template <typename PassManager, typename Pass>
  inline void setupPassManager(PassManager* passManager, Pass* pass) {
    passManager->add(new llvm::ProfileSummaryInfoWrapperPass());
    passManager->add(new llvm::TargetLibraryInfoWrapperPass(tlii()));
    passManager->add(createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));
    passManager->add(pass);
  }

  // Immutable state.
  const programl::ProgramGraphOptions programlOptions_;
  // Mutable state.
  std::unique_ptr<Benchmark> benchmark_;
  LlvmActionSpace actionSpace_;
  llvm::TargetLibraryInfoImpl tlii_;
};

}  // namespace compiler_gym::llvm_service

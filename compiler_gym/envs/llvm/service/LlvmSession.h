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
class LlvmSession {
 public:
  // Construct an environment by taking ownership of a benchmark. Throws
  // std::invalid_argument if the benchmark's LLVM module fails verification.
  LlvmSession(std::unique_ptr<Benchmark> benchmark, LlvmActionSpace actionSpace,
              const boost::filesystem::path& workingDirectory);

  inline const Benchmark& benchmark() const { return *benchmark_; }
  inline Benchmark& benchmark() { return *benchmark_; }

  inline const LlvmActionSpace actionSpace() const { return actionSpace_; }

  inline const boost::filesystem::path& workingDirectory() const { return workingDirectory_; }

  // Run the requested action(s) then compute the requested observation(s).
  [[nodiscard]] grpc::Status step(const StepRequest& request, StepReply* reply);

  // Returns the number of actions that have been applied in calls to step()
  // since the start of the session. This is just for logging and has no effect.
  inline int actionCount() const { return actionCount_; }

  // Run the requested action.
  [[nodiscard]] grpc::Status runAction(LlvmAction action, StepReply* reply);

  // Compute the requested observation.
  [[nodiscard]] grpc::Status getObservation(LlvmObservationSpace space, Observation* reply);

 protected:
  // Run the given pass, possibly modifying the underlying LLVM module.
  void runPass(llvm::Pass* pass, StepReply* reply);
  void runPass(llvm::FunctionPass* pass, StepReply* reply);

  // Run the commandline `opt` tool on the current LLVM module with the given
  // arguments, replacing the environment state with the generated output.
  [[nodiscard]] grpc::Status runOptWithArgs(const std::vector<std::string>& optArgs);

  inline const llvm::TargetLibraryInfoImpl& tlii() const { return tlii_; }

 private:
  // Setup pass manager with depdendent passes and the specified pass.
  template <typename PassManager, typename Pass>
  inline void setupPassManager(PassManager* passManager, Pass* pass) {
    passManager->add(new llvm::ProfileSummaryInfoWrapperPass());
    passManager->add(new llvm::TargetLibraryInfoWrapperPass(tlii()));
    passManager->add(createTargetTransformInfoWrapperPass(llvm::TargetIRAnalysis()));
    passManager->add(pass);
  }

  const boost::filesystem::path workingDirectory_;
  const std::unique_ptr<Benchmark> benchmark_;
  const LlvmActionSpace actionSpace_;
  const llvm::TargetLibraryInfoImpl tlii_;
  const programl::ProgramGraphOptions programlOptions_;

  int actionCount_;
};

}  // namespace compiler_gym::llvm_service

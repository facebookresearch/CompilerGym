// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <optional>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym {

// Base class for encapsulating an incremental compilation session.
//
// To add support for a new compiler, subclass from this base and provide
// implementations of the abstract methods, then call
// createAndRunCompilerGymService() and parametrize it with your class type:
//
//     #include "compiler_gym/service/CompilationSession.h"
//     #include "compiler_gym/service/runtime/Runtime.h"
//
//     using namespace compiler_gym;
//
//     class MyCompilationSession final : public CompilationSession { ... }
//
//     int main(int argc, char** argv) {
//         runtime::createAndRunCompilerGymService<MyCompilationSession>();
//     }
//
class CompilationSession {
 public:
  // Get the compiler version.
  virtual std::string getCompilerVersion() const;

  // A list of action spaces describing the capabilities of the compiler.
  virtual std::vector<ActionSpace> getActionSpaces() const = 0;

  // A list of feature vectors that this compiler provides.
  virtual std::vector<ObservationSpace> getObservationSpaces() const = 0;

  // Start a CompilationSession. This will be called after construction and
  // before applyAction() or computeObservation(). This will only be called
  // once.
  [[nodiscard]] virtual grpc::Status init(const ActionSpace& actionSpace,
                                          const Benchmark& benchmark) = 0;

  // Initialize the state from another CompilerSession. This will be called
  // after construction and before applyAction() or computeObservation(). This
  // will only be called once.
  [[nodiscard]] virtual grpc::Status init(CompilationSession* other);

  // Apply an action.
  [[nodiscard]] virtual grpc::Status applyAction(const Action& action, bool& endOfEpisode,
                                                 std::optional<ActionSpace>& newActionSpace,
                                                 bool& actionHadNoEffect) = 0;

  // Compute an observation.
  [[nodiscard]] virtual grpc::Status computeObservation(const ObservationSpace& observationSpace,
                                                        Observation& observation) = 0;

  // Optional. This will be called after all applyAction() and
  // computeObservation() in a step. Use this method if you would like to
  // perform post-transform validation of compiler state.
  [[nodiscard]] virtual grpc::Status endOfStep(bool actionHadNoEffect, bool& endOfEpisode,
                                               std::optional<ActionSpace>& newActionSpace);

  CompilationSession(const boost::filesystem::path& workingDirectory);

  virtual ~CompilationSession() = default;

 protected:
  // Get the working directory, which is a local filesystem directory that this
  // CompilationSession can use to store temporary files such as build
  // artifacts.
  inline const boost::filesystem::path& workingDirectory() { return workingDirectory_; }

 private:
  const boost::filesystem::path workingDirectory_;
};

}  // namespace compiler_gym

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// An example implementation of the CompilerGymService interface.
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/core/Core.h"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::example_service {

// The representation of a compilation session.
class ExampleCompilationSession final : public CompilationSession {
 public:
  using CompilationSession::CompilationSession;

  std::string getCompilerVersion() const override;

  std::vector<ActionSpace> getActionSpaces() const override;

  ActionSpace getActionSpace() const override { return actionSpace_; };

  std::vector<ObservationSpace> getObservationSpaces() const override;

  [[nodiscard]] grpc::Status init(size_t actionSpaceIndex,
                                  const compiler_gym::Benchmark& benchmark) override;

  [[nodiscard]] grpc::Status init(CompilationSession* other) override;

  [[nodiscard]] grpc::Status applyAction(size_t actionIndex, bool* endOfEpisode,
                                         bool* actionSpaceChanged,
                                         bool* actionHadNoEffect) override;

  [[nodiscard]] grpc::Status setObservation(size_t observationSpaceIndex,
                                            Observation* observation) override;

 private:
  ActionSpace actionSpace_;
};

}  // namespace compiler_gym::example_service

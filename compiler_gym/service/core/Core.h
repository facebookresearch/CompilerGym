// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym {

class CompilationSession {
 public:
  CompilationSession(const boost::filesystem::path& workingDirectory)
      : workingDirectory_(workingDirectory){};

  virtual ~CompilationSession() = default;

  virtual std::string getCompilerVersion() const { return ""; }
  virtual std::vector<ActionSpace> getActionSpaces() const = 0;  // what your compiler can do
  virtual std::vector<ObservationSpace> getObservationSpaces() const = 0;  // features you provide

  virtual ActionSpace getActionSpace() const = 0;

  [[nodiscard]] virtual grpc::Status init(size_t actionSpaceIndex, const Benchmark& benchmark) = 0;

  [[nodiscard]] virtual grpc::Status init(CompilationSession* other);

  [[nodiscard]] virtual grpc::Status applyAction(size_t actionIndex, bool* endOfEpisode,
                                                 bool* actionSpaceChanged,
                                                 bool* actionHadNoEffect) = 0;  // apply an action

  // Optional.
  [[nodiscard]] virtual grpc::Status endOfStep(bool* endOfEpisode);

  // compute an observation
  [[nodiscard]] virtual grpc::Status setObservation(size_t observationSpaceIndex,
                                                    Observation* observation) = 0;

 protected:
  inline const boost::filesystem::path& workingDirectory() { return workingDirectory_; }

 private:
  const boost::filesystem::path workingDirectory_;
};

}  // namespace compiler_gym

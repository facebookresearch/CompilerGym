// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// An example implementation of the CompilerGymService interface.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::example_service {

// Forward declaration of helper class.
class ExampleCompilationSession;

// An example compiler service. This class implements all of the RPC endpoints
// of the CompilerGymService interface.
class ExampleService final : public CompilerGymService::Service {
 public:
  explicit ExampleService(const boost::filesystem::path& workingDirectory);

  // RPC endpoints.
  grpc::Status GetVersion(grpc::ServerContext* context, const GetVersionRequest* request,
                          GetVersionReply* reply) final override;

  grpc::Status GetSpaces(grpc::ServerContext* context, const GetSpacesRequest* request,
                         GetSpacesReply* reply) final override;

  grpc::Status StartEpisode(grpc::ServerContext* context, const StartEpisodeRequest* request,
                            StartEpisodeReply* reply) final override;

  grpc::Status EndEpisode(grpc::ServerContext* context, const EndEpisodeRequest* request,
                          EndEpisodeReply* reply) final override;

  grpc::Status Step(grpc::ServerContext* context, const StepRequest* request,
                    StepReply* reply) final override;

  grpc::Status GetBenchmarks(grpc::ServerContext* context, const GetBenchmarksRequest* request,
                             GetBenchmarksReply* reply) final override;

 private:
  [[nodiscard]] grpc::Status session(uint64_t id, ExampleCompilationSession** environment);

  const boost::filesystem::path workingDirectory_;

  // A single compiler service can support multiple concurrent episodes. This
  // maps session IDs to objects that represent the individual episodes.
  std::unordered_map<int, std::unique_ptr<ExampleCompilationSession>> sessions_;

  std::vector<std::string> benchmarkNameList_;
  uint64_t nextSessionId_;
};

// The representation of a compilation session.
class ExampleCompilationSession {
 public:
  ExampleCompilationSession(const std::string& benchmark, ActionSpace actionSpace);

  [[nodiscard]] grpc::Status Step(const StepRequest* request, StepReply* reply);

 private:
  grpc::Status getObservation(int32_t observationSpace, Observation* reply);

  const std::string benchmark_;
  ActionSpace actionSpace_;
};

// Helper functions to describe the available action/observation/reward spaces.
std::vector<ActionSpace> getActionSpaces();
std::vector<ObservationSpace> getObservationSpaces();

}  // namespace compiler_gym::example_service

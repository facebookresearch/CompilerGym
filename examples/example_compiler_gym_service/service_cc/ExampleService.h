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

  grpc::Status StartSession(grpc::ServerContext* context, const StartSessionRequest* request,
                            StartSessionReply* reply) final override;

  grpc::Status EndSession(grpc::ServerContext* context, const EndSessionRequest* request,
                          EndSessionReply* reply) final override;

  grpc::Status Step(grpc::ServerContext* context, const StepRequest* request,
                    StepReply* reply) final override;

 private:
  [[nodiscard]] grpc::Status session(uint64_t id, ExampleCompilationSession** environment);

  const boost::filesystem::path workingDirectory_;

  // A single compiler service can support multiple concurrent sessions. This
  // maps session IDs to objects that represent the individual sessions.
  std::unordered_map<int, std::unique_ptr<ExampleCompilationSession>> sessions_;
  // Mutex used to ensure thread safety of creation and destruction of sessions.
  std::mutex sessionsMutex_;

  std::vector<std::string> benchmarkNameList_;
  uint64_t nextSessionId_;
};

// The representation of a compilation session.
class ExampleCompilationSession {
 public:
  ExampleCompilationSession(const std::string& benchmark, ActionSpace actionSpace);

  [[nodiscard]] grpc::Status Step(const StepRequest* request, StepReply* reply);

  grpc::Status getObservation(int32_t observationSpace, Observation* reply);

 private:
  const std::string benchmark_;
  ActionSpace actionSpace_;
};

}  // namespace compiler_gym::example_service

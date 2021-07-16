// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <mutex>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/CompilationSession.h"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/service/runtime/BenchmarkCache.h"

namespace compiler_gym::runtime {

/**
 * A default implementation of the CompilerGymService.
 *
 * When parametrized by a CompilationSession subclass, this provides the RPC
 * handling logic to run a gym service. User should call
 * createAndRunCompilerGymService() rather than interacting with this class
 * directly.
 */
template <typename CompilationSessionType>
class CompilerGymService final : public compiler_gym::CompilerGymService::Service {
 public:
  CompilerGymService(const boost::filesystem::path& workingDirectory,
                     std::unique_ptr<BenchmarkCache> benchmarks = nullptr);

  // RPC endpoints.
  grpc::Status GetVersion(grpc::ServerContext* context, const GetVersionRequest* request,
                          GetVersionReply* reply) final override;

  grpc::Status GetSpaces(grpc::ServerContext* context, const GetSpacesRequest* request,
                         GetSpacesReply* reply) final override;

  grpc::Status StartSession(grpc::ServerContext* context, const StartSessionRequest* request,
                            StartSessionReply* reply) final override;

  grpc::Status ForkSession(grpc::ServerContext* context, const ForkSessionRequest* request,
                           ForkSessionReply* reply) final override;

  grpc::Status EndSession(grpc::ServerContext* context, const EndSessionRequest* request,
                          EndSessionReply* reply) final override;

  // NOTE: Step() is not thread safe. The underlying assumption is that each
  // CompilationSessionType is managed by a single thread, so race conditions
  // between operations that affect the same CompilationSessionType are not
  // protected against.
  grpc::Status Step(grpc::ServerContext* context, const StepRequest* request,
                    StepReply* reply) final override;

  grpc::Status AddBenchmark(grpc::ServerContext* context, const AddBenchmarkRequest* request,
                            AddBenchmarkReply* reply) final override;

  grpc::Status SendSessionParameter(grpc::ServerContext* context,
                                    const SendSessionParameterRequest* request,
                                    SendSessionParameterReply* reply) final override;

  inline BenchmarkCache& benchmarks() { return *benchmarks_; }

  // Get the number of active sessions.
  inline int sessionCount() const { return static_cast<int>(sessions_.size()); }

 protected:
  [[nodiscard]] grpc::Status session(uint64_t id, CompilationSession** environment);

  [[nodiscard]] grpc::Status session(uint64_t id, const CompilationSession** environment) const;

  [[nodiscard]] grpc::Status action_space(const CompilationSession* session, int index,
                                          const ActionSpace** actionSpace) const;

  [[nodiscard]] grpc::Status observation_space(const CompilationSession* session, int index,
                                               const ObservationSpace** observationSpace) const;

  inline const boost::filesystem::path& workingDirectory() const { return workingDirectory_; }

  // Add the given session and return its ID.
  uint64_t addSession(std::unique_ptr<CompilationSession> session);

  // Handle a built-in session parameter.
  [[nodiscard]] grpc::Status handleBuiltinSessionParameter(const std::string& key,
                                                           const std::string& value,
                                                           std::optional<std::string>& reply);

 private:
  const boost::filesystem::path workingDirectory_;
  const std::vector<ActionSpace> actionSpaces_;
  const std::vector<ObservationSpace> observationSpaces_;

  std::unordered_map<uint64_t, std::unique_ptr<CompilationSession>> sessions_;
  std::unique_ptr<BenchmarkCache> benchmarks_;

  // Mutex used to ensure thread safety of creation and destruction of sessions.
  std::mutex sessionsMutex_;
  uint64_t nextSessionId_;
};

}  // namespace compiler_gym::runtime

#include "compiler_gym/service/runtime/CompilerGymServiceImpl.h"

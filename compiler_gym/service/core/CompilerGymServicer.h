// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <fmt/format.h>
#include <grpcpp/grpcpp.h>

#include <memory>
#include <mutex>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/core/BenchmarkProtoCache.h"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/Version.h"

namespace compiler_gym {

template <typename CompilationSession>
class CompilerGymServicer final : public CompilerGymService::Service {
 public:
  explicit CompilerGymServicer(const boost::filesystem::path& workingDirectory)
      : workingDirectory_(workingDirectory) {}

  // RPC endpoints.
  grpc::Status GetVersion(grpc::ServerContext* context, const GetVersionRequest* request,
                          GetVersionReply* reply) final override {
    VLOG(2) << "GetSpaces()";
    reply->set_service_version(COMPILER_GYM_VERSION);
    CompilationSession session(workingDirectory());
    reply->set_compiler_version(session.getCompilerVersion());
    return grpc::Status::OK;
  }

  grpc::Status GetSpaces(grpc::ServerContext* context, const GetSpacesRequest* request,
                         GetSpacesReply* reply) final override {
    VLOG(2) << "GetSpaces()";
    CompilationSession session(workingDirectory());
    const auto actionSpaces = session.getActionSpaces();
    *reply->mutable_action_space_list() = {actionSpaces.begin(), actionSpaces.end()};
    const auto observationSpaces = session.getObservationSpaces();
    *reply->mutable_observation_space_list() = {observationSpaces.begin(), observationSpaces.end()};
    return grpc::Status::OK;
  }

  grpc::Status StartSession(grpc::ServerContext* context, const StartSessionRequest* request,
                            StartSessionReply* reply) final override {
    if (!request->benchmark().size()) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "No benchmark URI set for StartSession()");
    }

    VLOG(1) << "StartSession(" << request->benchmark() << "), [" << nextSessionId_ << "]";
    const std::lock_guard<std::mutex> lock(sessionsMutex_);

    Benchmark benchmark;
    RETURN_IF_ERROR(benchmarkCache_.getBenchmark(request->benchmark(), &benchmark));

    // Construct the new session.
    auto session = std::make_unique<CompilationSession>(workingDirectory());

    // Compute the initial observations.
    for (int i = 0; i < request->observation_space_size(); ++i) {
      RETURN_IF_ERROR(
          session->setObservation(request->observation_space(i), reply->add_observation()));
    }

    reply->set_session_id(nextSessionId_);
    sessions_[nextSessionId_] = std::move(session);
    ++nextSessionId_;

    return grpc::Status::OK;
  }

  grpc::Status ForkSession(grpc::ServerContext* context, const ForkSessionRequest* request,
                           ForkSessionReply* reply) final override {
    const std::lock_guard<std::mutex> lock(sessionsMutex_);

    CompilationSession* baseSession;
    RETURN_IF_ERROR(session(request->session_id(), &baseSession));
    VLOG(1) << "ForkSession(" << request->session_id() << "), [" << nextSessionId_ << "]";

    // Construct the new session.
    auto forked = std::make_unique<CompilationSession>(workingDirectory());

    // Initialize from the base environment.
    RETURN_IF_ERROR(forked->init(*baseSession));

    reply->set_session_id(nextSessionId_);
    sessions_[nextSessionId_] = std::move(forked);
    ++nextSessionId_;

    return grpc::Status::OK;
  }

  grpc::Status EndSession(grpc::ServerContext* context, const EndSessionRequest* request,
                          EndSessionReply* reply) final override {
    const std::lock_guard<std::mutex> lock(sessionsMutex_);

    // Note that unlike the other methods, no error is thrown if the requested
    // session does not exist.
    if (sessions_.find(request->session_id()) != sessions_.end()) {
      const CompilationSession* environment;
      RETURN_IF_ERROR(session(request->session_id(), &environment));
      sessions_.erase(request->session_id());
      VLOG(1) << "EndSession(" << request->session_id() << "), " << sessions_.size()
              << " remaining";
    }

    reply->set_remaining_sessions(sessions_.size());
    return Status::OK;
  }

  // NOTE: Step() is not thread safe. The underlying assumption is that each
  // CompilationSession is managed by a single thread, so race conditions
  // between operations that affect the same CompilationSession are not
  // protected against.
  grpc::Status Step(grpc::ServerContext* context, const StepRequest* request,
                    StepReply* reply) final override {
    CompilationSession* environment;
    RETURN_IF_ERROR(session(request->session_id(), &environment));

    VLOG(2) << "Session " << request->session_id() << " Step()";

    // Apply the actions.
    bool endOfEpisode = false;
    bool newActionSpace = false;
    bool actionsHadNoEffect = false;
    for (int i = 0; i < request->action_size(); ++i) {
      bool actionHadNoEffect = false;
      RETURN_IF_ERROR(environment->applyAction(request->action(i), &endOfEpisode, &newActionSpace,
                                               &actionHadNoEffect));
      actionsHadNoEffect &= actionHadNoEffect;
      if (endOfEpisode) {
        reply->set_end_of_session(true);
        break;
      }
    }

    reply->set_action_had_no_effect(actionsHadNoEffect);
    if (newActionSpace) {
      *reply->mutable_new_action_space() = environment->getActionSpace();
    }

    // Compute the requested observations.
    for (int i = 0; i < request->observation_space_size(); ++i) {
      RETURN_IF_ERROR(
          environment->setObservation(request->observation_space(i), reply->add_observation()));
    }

    return grpc::Status::OK;
  }

  grpc::Status AddBenchmark(grpc::ServerContext* context, const AddBenchmarkRequest* request,
                            AddBenchmarkReply* reply) final override {
    // We need to grab the sessions lock here as benchmarkCache_ is not thread
    // safe and the only place it is touched is StartSession().
    const std::lock_guard<std::mutex> lock(sessionsMutex_);

    VLOG(2) << "AddBenchmark()";
    for (int i = 0; i < request->benchmark_size(); ++i) {
      RETURN_IF_ERROR(benchmarkCache_.addBenchmark(std::move(request->benchmark(i))));
    }

    return grpc::Status::OK;
  }

 protected:
  grpc::Status session(uint64_t id, CompilationSession** environment) {
    auto it = sessions_.find(id);
    if (it == sessions_.end()) {
      return Status(grpc::StatusCode::NOT_FOUND, fmt::format("Session not found: {}", id));
    }

    *environment = it->second.get();
    return grpc::Status::OK;
  }

  grpc::Status session(uint64_t id, const CompilationSession** environment) const {
    auto it = sessions_.find(id);
    if (it == sessions_.end()) {
      return grpc::Status(grpc::StatusCode::NOT_FOUND, fmt::format("Session not found: {}", id));
    }

    *environment = it->second.get();
    return grpc::Status::OK;
  }

  inline const boost::filesystem::path& workingDirectory() const { return workingDirectory_; }

 private:
  const boost::filesystem::path workingDirectory_;

  std::unordered_map<uint64_t, std::unique_ptr<CompilationSession>> sessions_;
  BenchmarkProtoCache benchmarkCache_;

  // Mutex used to ensure thread safety of creation and destruction of sessions.
  std::mutex sessionsMutex_;
  uint64_t nextSessionId_;
};

}  // namespace compiler_gym

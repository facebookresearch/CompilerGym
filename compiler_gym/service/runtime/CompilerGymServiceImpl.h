// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.
//
// Private implementation of the CompilerGymService template class. Do not
// include this header directly! Use
// compiler_gym/service/runtimeCompilerGymService.h.
#pragma once

#include <fmt/format.h>

#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/Version.h"

namespace compiler_gym::runtime {

template <typename CompilationSessionType>
CompilerGymService<CompilationSessionType>::CompilerGymService(
    const boost::filesystem::path& workingDirectory, std::unique_ptr<BenchmarkCache> benchmarks)
    : workingDirectory_(workingDirectory),
      actionSpaces_(CompilationSessionType(workingDirectory).getActionSpaces()),
      observationSpaces_(CompilationSessionType(workingDirectory).getObservationSpaces()),
      benchmarks_(benchmarks ? std::move(benchmarks) : std::make_unique<BenchmarkCache>()),
      nextSessionId_(0) {}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::GetVersion(
    grpc::ServerContext* context, const GetVersionRequest* request, GetVersionReply* reply) {
  VLOG(2) << "GetVersion()";
  reply->set_service_version(COMPILER_GYM_VERSION);
  CompilationSessionType environment(workingDirectory());
  reply->set_compiler_version(environment.getCompilerVersion());
  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::GetSpaces(grpc::ServerContext* context,
                                                                   const GetSpacesRequest* request,
                                                                   GetSpacesReply* reply) {
  VLOG(2) << "GetSpaces()";
  for (const auto& actionSpace : actionSpaces_) {
    *reply->add_action_space_list() = actionSpace;
  }
  for (const auto& observationSpace : observationSpaces_) {
    *reply->add_observation_space_list() = observationSpace;
  }
  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::StartSession(
    grpc::ServerContext* context, const StartSessionRequest* request, StartSessionReply* reply) {
  if (!request->benchmark().uri().size()) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        "No benchmark URI set for StartSession()");
  }

  const std::lock_guard<std::mutex> lock(sessionsMutex_);
  VLOG(1) << "StartSession(id=" << nextSessionId_ << ", benchmark=" << request->benchmark().uri()
          << "), " << (sessionCount() + 1) << " active sessions";

  // If a benchmark definition was provided, add it.
  if (request->benchmark().has_program()) {
    benchmarks().add(std::move(request->benchmark()));
  }

  // Lookup the requested benchmark.
  const Benchmark* benchmark = benchmarks().get(request->benchmark().uri());
  if (!benchmark) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND, "Benchmark not found");
  }

  // Construct the new session.
  auto environment = std::make_unique<CompilationSessionType>(workingDirectory());

  // Resolve the action space.
  const ActionSpace* actionSpace;
  RETURN_IF_ERROR(action_space(environment.get(), request->action_space(), &actionSpace));

  // Initialize the session.
  RETURN_IF_ERROR(environment->init(*actionSpace, *benchmark));

  // Compute the initial observations.
  for (int i = 0; i < request->observation_space_size(); ++i) {
    const ObservationSpace* observationSpace;
    RETURN_IF_ERROR(
        observation_space(environment.get(), request->observation_space(i), &observationSpace));
    RETURN_IF_ERROR(environment->computeObservation(*observationSpace, *reply->add_observation()));
  }

  reply->set_session_id(addSession(std::move(environment)));

  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::ForkSession(
    grpc::ServerContext* context, const ForkSessionRequest* request, ForkSessionReply* reply) {
  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  CompilationSession* baseSession;
  RETURN_IF_ERROR(session(request->session_id(), &baseSession));
  VLOG(1) << "ForkSession(" << request->session_id() << "), [" << nextSessionId_ << "]";

  // Construct the new session.
  auto forked = std::make_unique<CompilationSessionType>(workingDirectory());

  // Initialize from the base environment.
  RETURN_IF_ERROR(forked->init(baseSession));

  reply->set_session_id(addSession(std::move(forked)));

  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::EndSession(
    grpc::ServerContext* context, const EndSessionRequest* request, EndSessionReply* reply) {
  VLOG(1) << "EndSession(id=" << request->session_id() << "), " << sessionCount() - 1
          << " sessions remaining";

  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  // Note that unlike the other methods, no error is thrown if the requested
  // session does not exist.
  if (sessions_.find(request->session_id()) != sessions_.end()) {
    const CompilationSession* environment;
    RETURN_IF_ERROR(session(request->session_id(), &environment));
    sessions_.erase(request->session_id());
  }

  reply->set_remaining_sessions(sessionCount());
  return Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::Step(grpc::ServerContext* context,
                                                              const StepRequest* request,
                                                              StepReply* reply) {
  CompilationSession* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));

  VLOG(2) << "Session " << request->session_id() << " Step()";

  bool endOfEpisode = false;
  std::optional<ActionSpace> newActionSpace;
  bool actionsHadNoEffect = true;

  // Apply the actions.
  for (int i = 0; i < request->action_size(); ++i) {
    bool actionHadNoEffect = false;
    std::optional<ActionSpace> newActionSpaceFromAction;
    RETURN_IF_ERROR(environment->applyAction(request->action(i), endOfEpisode,
                                             newActionSpaceFromAction, actionHadNoEffect));
    actionsHadNoEffect &= actionHadNoEffect;
    if (newActionSpaceFromAction.has_value()) {
      newActionSpace = *newActionSpaceFromAction;
    }
    if (endOfEpisode) {
      break;
    }
  }

  // Compute the requested observations.
  for (int i = 0; i < request->observation_space_size(); ++i) {
    const ObservationSpace* observationSpace;
    RETURN_IF_ERROR(
        observation_space(environment, request->observation_space(i), &observationSpace));
    DCHECK(observationSpace) << "No observation space set";
    RETURN_IF_ERROR(environment->computeObservation(*observationSpace, *reply->add_observation()));
  }

  // Call the end-of-step callback.
  RETURN_IF_ERROR(environment->endOfStep(actionsHadNoEffect, endOfEpisode, newActionSpace));

  reply->set_action_had_no_effect(actionsHadNoEffect);
  if (newActionSpace.has_value()) {
    *reply->mutable_new_action_space() = *newActionSpace;
  }
  reply->set_end_of_session(endOfEpisode);
  return Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::AddBenchmark(
    grpc::ServerContext* context, const AddBenchmarkRequest* request, AddBenchmarkReply* reply) {
  // We need to grab the sessions lock here to ensure thread safe access to the
  // benchmarks cache.
  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  VLOG(2) << "AddBenchmark()";
  for (int i = 0; i < request->benchmark_size(); ++i) {
    benchmarks().add(std::move(request->benchmark(i)));
  }

  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::SendSessionParameter(
    grpc::ServerContext* context, const SendSessionParameterRequest* request,
    SendSessionParameterReply* reply) {
  CompilationSession* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));

  VLOG(2) << "Session " << request->session_id() << " SendSessionParameter()";

  for (int i = 0; i < request->parameter_size(); ++i) {
    const auto& param = request->parameter(i);
    std::optional<std::string> message{std::nullopt};

    // Handle each parameter in the session and generate a response.
    RETURN_IF_ERROR(environment->handleSessionParameter(param.key(), param.value(), message));

    // Use the builtin parameter handlers if not handled by a session.
    if (!message.has_value()) {
      RETURN_IF_ERROR(handleBuiltinSessionParameter(param.key(), param.value(), message));
    }

    if (message.has_value()) {
      *reply->add_reply() = *message;
    } else {
      return Status(grpc::StatusCode::INVALID_ARGUMENT,
                    fmt::format("Unknown parameter: {}", param.key()));
    }
  }

  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::handleBuiltinSessionParameter(
    const std::string& key, const std::string& value, std::optional<std::string>& reply) {
  if (key == "service.benchmark_cache.set_max_size_in_bytes") {
    benchmarks().setMaxSizeInBytes(std::stoi(value));
    reply = value;
  } else if (key == "service.benchmark_cache.get_max_size_in_bytes") {
    reply = fmt::format("{}", benchmarks().maxSizeInBytes());
  } else if (key == "service.benchmark_cache.get_size_in_bytes") {
    reply = fmt::format("{}", benchmarks().sizeInBytes());
  }

  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::session(uint64_t id,
                                                                 CompilationSession** environment) {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return Status(grpc::StatusCode::NOT_FOUND, fmt::format("Session not found: {}", id));
  }

  *environment = it->second.get();
  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::session(
    uint64_t id, const CompilationSession** environment) const {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return grpc::Status(grpc::StatusCode::NOT_FOUND, fmt::format("Session not found: {}", id));
  }

  *environment = it->second.get();
  return grpc::Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::action_space(
    const CompilationSession* session, int index, const ActionSpace** actionSpace) const {
  if (index < 0 || index >= static_cast<int>(actionSpaces_.size())) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        fmt::format("Action space index out of range: {}", index));
  }
  *actionSpace = &actionSpaces_[index];
  return Status::OK;
}

template <typename CompilationSessionType>
grpc::Status CompilerGymService<CompilationSessionType>::observation_space(
    const CompilationSession* session, int index, const ObservationSpace** observationSpace) const {
  if (index < 0 || index >= static_cast<int>(observationSpaces_.size())) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        fmt::format("Observation space index out of range: {}", index));
  }
  *observationSpace = &observationSpaces_[index];
  return Status::OK;
}

template <typename CompilationSessionType>
uint64_t CompilerGymService<CompilationSessionType>::addSession(
    std::unique_ptr<CompilationSession> session) {
  uint64_t id = nextSessionId_;
  sessions_[id] = std::move(session);
  ++nextSessionId_;
  return id;
}

}  // namespace compiler_gym::runtime

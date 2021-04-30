// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "examples/example_compiler_gym_service/service_cc/ExampleService.h"

#include <fmt/format.h>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/Version.h"

namespace compiler_gym::example_service {

using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

namespace fs = boost::filesystem;

namespace {

template <typename T>
[[nodiscard]] inline Status rangeCheck(const T& value, const T& minValue, const T& maxValue) {
  if (value < minValue || value > maxValue) {
    return Status(StatusCode::INVALID_ARGUMENT, "Out-of-range");
  }
  return Status::OK;
}

std::vector<std::string> getBenchmarkUris() {
  return {"benchmark://example-v0/foo", "benchmark://example-v0/bar"};
}

std::vector<ActionSpace> getActionSpaces() {
  ActionSpace space;
  space.set_name("default");
  space.add_action("a");
  space.add_action("b");
  space.add_action("c");

  return {space};
}

std::vector<ObservationSpace> getObservationSpaces() {
  ObservationSpace ir;
  ir.set_name("ir");
  ScalarRange irSizeRange;
  irSizeRange.mutable_min()->set_value(0);
  *ir.mutable_string_size_range() = irSizeRange;
  ir.set_deterministic(true);
  ir.set_platform_dependent(false);

  ObservationSpace features;
  features.set_name("features");
  for (int i = 0; i < 3; ++i) {
    ScalarRange* featureSizeRange = features.mutable_int64_range_list()->add_range();
    featureSizeRange->mutable_min()->set_value(-100);
    featureSizeRange->mutable_max()->set_value(100);
  }

  ObservationSpace runtime;
  runtime.set_name("runtime");
  ScalarRange runtimeRange;
  runtimeRange.mutable_min()->set_value(0);
  *runtime.mutable_scalar_double_range() = runtimeRange;
  runtime.set_deterministic(false);
  runtime.set_platform_dependent(true);

  return {ir, features, runtime};
}

}  // namespace

ExampleService::ExampleService(const fs::path& workingDirectory)
    : workingDirectory_(workingDirectory), nextSessionId_(0) {}

Status ExampleService::GetVersion(ServerContext* /* unused */,
                                  const GetVersionRequest* /* unused */, GetVersionReply* reply) {
  reply->set_service_version(COMPILER_GYM_VERSION);
  reply->set_compiler_version("1.0.0");
  return Status::OK;
}

Status ExampleService::GetSpaces(ServerContext* /* unused*/, const GetSpacesRequest* /* unused */,
                                 GetSpacesReply* reply) {
  const auto actionSpaces = getActionSpaces();
  const auto observationSpaces = getObservationSpaces();

  *reply->mutable_action_space_list() = {actionSpaces.begin(), actionSpaces.end()};
  *reply->mutable_observation_space_list() = {observationSpaces.begin(), observationSpaces.end()};
  return Status::OK;
}

Status ExampleService::StartSession(ServerContext* /* unused*/, const StartSessionRequest* request,
                                    StartSessionReply* reply) {
  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  // Determine the benchmark to use.
  std::string benchmark = request->benchmark();
  const auto benchmarks = getBenchmarkUris();
  if (!benchmark.empty() &&
      std::find(benchmarks.begin(), benchmarks.end(), benchmark) == benchmarks.end()) {
    return Status(StatusCode::INVALID_ARGUMENT, "Unknown program name");
  } else {
    // If no benchmark was requested, choose one.
    benchmark = "foo";
  }
  reply->set_benchmark(benchmark);

  // Determine the action space.
  const auto actionSpaces = getActionSpaces();
  RETURN_IF_ERROR(
      rangeCheck(request->action_space(), 0, static_cast<int32_t>(actionSpaces.size()) - 1));
  const auto actionSpace = actionSpaces[request->action_space()];

  // Create the new compilation session given.
  auto session = std::make_unique<ExampleCompilationSession>(benchmark, actionSpace);

  // Generate initial observations.
  for (int i = 0; i < request->observation_space_size(); ++i) {
    RETURN_IF_ERROR(rangeCheck(request->observation_space(i), 0,
                               static_cast<int32_t>(getObservationSpaces().size()) - 1));
    RETURN_IF_ERROR(
        session->getObservation(request->observation_space(i), reply->add_observation()));
  }

  reply->set_session_id(nextSessionId_);
  sessions_[nextSessionId_] = std::move(session);
  ++nextSessionId_;

  return Status::OK;
}

Status ExampleService::EndSession(ServerContext* /* unused*/, const EndSessionRequest* request,
                                  EndSessionReply* /* unused */) {
  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  auto session = sessions_.find(request->session_id());
  // De-allocate the session.
  if (session != sessions_.end()) {
    sessions_.erase(session);
  }
  return Status::OK;
}

Status ExampleService::Step(ServerContext* /* unused*/, const StepRequest* request,
                            StepReply* reply) {
  ExampleCompilationSession* sess;
  RETURN_IF_ERROR(session(request->session_id(), &sess));
  return sess->Step(request, reply);
}

Status ExampleService::session(uint64_t id, ExampleCompilationSession** sess) {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Session not found: {}", id));
  }
  *sess = it->second.get();
  return Status::OK;
}

ExampleCompilationSession::ExampleCompilationSession(const std::string& benchmark,
                                                     ActionSpace actionSpace)
    : benchmark_(benchmark), actionSpace_(actionSpace) {}

Status ExampleCompilationSession::Step(const StepRequest* request, StepReply* reply) {
  for (int i = 0; i < request->action_size(); ++i) {
    const auto action = request->action(i);
    // Run the actual action. Here we just range check.
    RETURN_IF_ERROR(rangeCheck(action, 0, static_cast<int32_t>(actionSpace_.action_size() - 1)));
  }

  // Generate observations.
  for (int i = 0; i < request->observation_space_size(); ++i) {
    RETURN_IF_ERROR(rangeCheck(request->observation_space(i), 0,
                               static_cast<int32_t>(getObservationSpaces().size()) - 1));
    RETURN_IF_ERROR(getObservation(request->observation_space(i), reply->add_observation()));
  }

  return Status::OK;
}

Status ExampleCompilationSession::getObservation(int32_t observationSpace,
                                                 Observation* observation) {
  const auto observationSpaces = getObservationSpaces();
  RETURN_IF_ERROR(
      rangeCheck(observationSpace, 0, static_cast<int32_t>(observationSpaces.size()) - 1));
  switch (observationSpace) {
    case 0:  // IR
      observation->set_string_value("Hello, world!");
      break;
    case 1:  // Features
      for (int i = 0; i < 3; ++i) {
        observation->mutable_int64_list()->add_value(0);
      }
      break;
    case 2:  // Runtime
      observation->set_scalar_double(0);
    default:
      break;
  }
  return Status::OK;
}

}  // namespace compiler_gym::example_service

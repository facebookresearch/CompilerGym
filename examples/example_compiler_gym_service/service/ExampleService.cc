// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "examples/example_compiler_gym_service/service/ExampleService.h"

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

}  // namespace

std::vector<std::string> getBenchmarks() { return {"foo", "bar"}; }

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

  ObservationSpace features;
  features.set_name("features");
  for (int i = 0; i < 3; ++i) {
    ScalarRange* featureSizeRange = features.mutable_int64_range_list()->add_range();
    featureSizeRange->mutable_min()->set_value(-100);
    featureSizeRange->mutable_max()->set_value(100);
  }

  return {ir, features};
}

std::vector<RewardSpace> getRewardSpaces() {
  RewardSpace codesize;
  codesize.set_name("codesize");
  codesize.mutable_range()->mutable_max()->set_value(0);

  return {codesize};
}

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
  const auto rewardSpaces = getRewardSpaces();

  *reply->mutable_action_space_list() = {actionSpaces.begin(), actionSpaces.end()};
  *reply->mutable_observation_space_list() = {observationSpaces.begin(), observationSpaces.end()};
  *reply->mutable_reward_space_list() = {rewardSpaces.begin(), rewardSpaces.end()};
  return Status::OK;
}

Status ExampleService::StartEpisode(ServerContext* /* unused*/, const StartEpisodeRequest* request,
                                    StartEpisodeReply* reply) {
  // Determine the benchmark to use.
  std::string benchmark = request->benchmark();
  const auto benchmarks = getBenchmarks();
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

  // Range check the eager observation space.
  std::optional<int32_t> eagerObservation = std::nullopt;
  if (request->use_eager_observation_space()) {
    eagerObservation = request->eager_observation_space();
    RETURN_IF_ERROR(
        rangeCheck(*eagerObservation, 0, static_cast<int32_t>(getObservationSpaces().size()) - 1));
  }

  // Range check the eager reward space.
  std::optional<int32_t> eagerReward = std::nullopt;
  if (request->use_eager_reward_space()) {
    eagerReward = request->eager_reward_space();
    RETURN_IF_ERROR(
        rangeCheck(*eagerReward, 0, static_cast<int32_t>(getRewardSpaces().size()) - 1));
  }

  // Create the new compilation session given.
  reply->set_session_id(nextSessionId_);
  sessions_[nextSessionId_] = std::make_unique<ExampleCompilationSession>(
      benchmark, actionSpace, eagerObservation, eagerReward);
  ++nextSessionId_;

  return Status::OK;
}

Status ExampleService::EndEpisode(ServerContext* /* unused*/, const EndEpisodeRequest* request,
                                  EndEpisodeReply* /* unused */) {
  auto session = sessions_.find(request->session_id());
  // De-allocate the session.
  if (session != sessions_.end()) {
    sessions_.erase(session);
  }
  return Status::OK;
}

Status ExampleService::TakeAction(ServerContext* /* unused*/, const ActionRequest* request,
                                  ActionReply* reply) {
  ExampleCompilationSession* sess;
  RETURN_IF_ERROR(session(request->session_id(), &sess));
  return sess->takeAction(request, reply);
}

Status ExampleService::GetObservation(ServerContext* /* unused*/, const ObservationRequest* request,
                                      Observation* reply) {
  ExampleCompilationSession* sess;
  RETURN_IF_ERROR(session(request->session_id(), &sess));
  return sess->getObservation(request->observation_space(), reply);
}

Status ExampleService::GetReward(ServerContext* /* unused*/, const RewardRequest* request,
                                 Reward* reply) {
  ExampleCompilationSession* sess;
  RETURN_IF_ERROR(session(request->session_id(), &sess));
  return sess->getReward(request->reward_space(), reply);
}

Status ExampleService::GetBenchmarks(grpc::ServerContext* /*unused*/,
                                     const GetBenchmarksRequest* /*unused*/,
                                     GetBenchmarksReply* reply) {
  const auto benchmarks = getBenchmarks();
  *reply->mutable_benchmark() = {benchmarks.begin(), benchmarks.end()};
  return Status::OK;
}

Status ExampleService::session(uint64_t id, ExampleCompilationSession** sess) {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return Status(StatusCode::INVALID_ARGUMENT, "Session ID not found");
  }
  *sess = it->second.get();
  return Status::OK;
}

ExampleCompilationSession::ExampleCompilationSession(const std::string& benchmark,
                                                     ActionSpace actionSpace,
                                                     std::optional<int32_t> eagerObservation,
                                                     std::optional<int32_t> eagerReward)
    : benchmark_(benchmark),
      actionSpace_(actionSpace),
      eagerObservation_(eagerObservation),
      eagerReward_(eagerReward) {}

Status ExampleCompilationSession::takeAction(const ActionRequest* request, ActionReply* reply) {
  for (int i = 0; i < request->action_size(); ++i) {
    const auto action = request->action(i);
    // Run the actual action. Here we just range check.
    RETURN_IF_ERROR(rangeCheck(action, 0, static_cast<int32_t>(actionSpace_.action_size() - 1)));
  }

  // Compute a new observation and reward if required.
  if (eagerObservation_.has_value()) {
    RETURN_IF_ERROR(getObservation(*eagerObservation_, reply->mutable_observation()));
  }
  if (eagerReward_.has_value()) {
    RETURN_IF_ERROR(getReward(*eagerReward_, reply->mutable_reward()));
  }
  return Status::OK;
}

Status ExampleCompilationSession::getObservation(int32_t observationSpace,
                                                 Observation* observation) {
  const auto observationSpaces = getObservationSpaces();
  RETURN_IF_ERROR(
      rangeCheck(observationSpace, 0, static_cast<int32_t>(observationSpaces.size()) - 1));
  switch (observationSpace) {
    case 0:
      observation->set_string_value("Hello, world!");
      break;
    case 1:
      for (int i = 0; i < 3; ++i) {
        observation->mutable_int64_list()->add_value(0);
      }
      break;
    default:
      break;
  }
  return Status::OK;
}

Status ExampleCompilationSession::getReward(int32_t rewardSpace, Reward* reward) {
  const auto rewardSpaces = getRewardSpaces();
  RETURN_IF_ERROR(rangeCheck(rewardSpace, 0, static_cast<int32_t>(rewardSpaces.size()) - 1));
  reward->set_reward(0);
  return Status::OK;
}

}  // namespace compiler_gym::example_service

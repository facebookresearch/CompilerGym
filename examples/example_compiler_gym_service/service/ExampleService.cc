// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "examples/example_compiler_gym_service/service/ExampleService.h"

#include <stdexcept>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/Version.h"

namespace compiler_gym::example_service {

using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;

namespace fs = boost::filesystem;

template <typename T>
[[nodiscard]] inline Status rangeCheck(const T& value, const T& minValue, const T& maxValue) {
  if (value < minValue || value > maxValue) {
    return Status(StatusCode::INVALID_ARGUMENT, "Out-of-range");
  }
  return Status::OK;
}

ExampleService::ExampleService(const fs::path& workingDirectory)
    : workingDirectory_(workingDirectory),
      episodeStarted_(false),
      programNameList_({"foo", "bar"}),
      actionSpace_({"a", "b", "c"}),
      eagerObservation_(false),
      eagerReward_(false) {
  ObservationSpace ir;
  ir.set_name("ir");
  ScalarRange irSizeRange;
  irSizeRange.mutable_min()->set_value(0);
  *ir.mutable_string_size_range() = irSizeRange;
  observationSpaces_.push_back(ir);

  ObservationSpace features;
  features.set_name("features");
  for (int i = 0; i < 3; ++i) {
    ScalarRange* featureSizeRange = features.mutable_int64_range_list()->add_range();
    featureSizeRange->mutable_min()->set_value(-100);
    featureSizeRange->mutable_max()->set_value(100);
  }
  observationSpaces_.push_back(features);

  RewardSpace codesize;
  codesize.set_name("codesize");
  codesize.mutable_range()->mutable_max()->set_value(0);
  rewardSpaces_.push_back(codesize);
}

Status ExampleService::GetVersion(ServerContext* /* unused */,
                                  const GetVersionRequest* /* unused */, GetVersionReply* reply) {
  reply->set_service_version(COMPILER_GYM_VERSION);
  reply->set_compiler_version("1.0.0");
  return Status::OK;
}

Status ExampleService::GetSpaces(ServerContext* /* unused*/, const GetSpacesRequest* /* unused */,
                                 GetSpacesReply* reply) {
  auto actionSpace = reply->add_action_space_list();
  actionSpace->set_name("default");
  *actionSpace->mutable_action() = {actionSpace_.begin(), actionSpace_.end()};
  *reply->mutable_observation_space_list() = {observationSpaces_.begin(), observationSpaces_.end()};
  *reply->mutable_reward_space_list() = {rewardSpaces_.begin(), rewardSpaces_.end()};

  return Status::OK;
}

Status ExampleService::StartEpisode(ServerContext* /* unused*/, const StartEpisodeRequest* request,
                                    StartEpisodeReply* reply) {
  if (episodeStarted_) {
    return Status(StatusCode::FAILED_PRECONDITION, "Already started episode");
  }

  // Program.
  std::string benchmark = request->benchmark();
  if (benchmark.size() && (benchmark != "foo" && benchmark != "bar")) {
    return Status(StatusCode::INVALID_ARGUMENT, "Unknown program name");
  } else {
    benchmark = "foo";  // Choose the benchmark to use.
  }

  reply->set_benchmark(benchmark);

  // Eager observation.
  if (request->use_eager_observation_space()) {
    eagerObservation_ = true;
    eagerObservationSpace_ = request->eager_observation_space();
    RETURN_IF_ERROR(
        rangeCheck(eagerObservationSpace_, 0, static_cast<int32_t>(observationSpaces_.size()) - 1));
  }

  // Eager reward.
  if (request->use_eager_reward_space()) {
    eagerReward_ = true;
    eagerRewardSpace_ = request->eager_reward_space();
    RETURN_IF_ERROR(
        rangeCheck(eagerRewardSpace_, 0, static_cast<int32_t>(rewardSpaces_.size()) - 1));
  }

  episodeStarted_ = true;

  return Status::OK;
}

Status ExampleService::EndEpisode(ServerContext* /* unused*/, const EndEpisodeRequest* /* unused */,
                                  EndEpisodeReply* /* unused */) {
  episodeStarted_ = false;
  return Status::OK;
}

Status ExampleService::TakeAction(ServerContext* /* unused*/, const ActionRequest* request,
                                  ActionReply* reply) {
  if (!episodeStarted_) {
    return Status(StatusCode::FAILED_PRECONDITION, "No episode");
  }

  for (int i = 0; i < request->action_size(); ++i) {
    const auto& action = request->action(i);
    RETURN_IF_ERROR(rangeCheck(action, 0, static_cast<int32_t>(actionSpace_.size()) - 1));
  }

  if (eagerObservation_) {
    RETURN_IF_ERROR(setObservation(eagerObservationSpace_, reply->mutable_observation()));
  }

  if (eagerReward_) {
    RETURN_IF_ERROR(setReward(eagerRewardSpace_, reply->mutable_reward()));
  }

  return Status::OK;
}

Status ExampleService::GetObservation(ServerContext* /* unused*/, const ObservationRequest* request,
                                      Observation* reply) {
  if (!episodeStarted_) {
    return Status(StatusCode::FAILED_PRECONDITION, "No episode");
  }
  return setObservation(request->observation_space(), reply);
}

Status ExampleService::GetReward(ServerContext* /* unused*/, const RewardRequest* request,
                                 Reward* reply) {
  if (!episodeStarted_) {
    return Status(StatusCode::FAILED_PRECONDITION, "No episode");
  }
  return setReward(request->reward_space(), reply);
}

Status ExampleService::GetBenchmarks(grpc::ServerContext* /*unused*/,
                                     const GetBenchmarksRequest* /*unused*/,
                                     GetBenchmarksReply* reply) {
  reply->add_benchmark("foo");
  reply->add_benchmark("bar");

  return Status::OK;
}

Status ExampleService::setObservation(int32_t observationSpace, Observation* observation) {
  RETURN_IF_ERROR(
      rangeCheck(observationSpace, 0, static_cast<int32_t>(observationSpaces_.size()) - 1));
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

Status ExampleService::setReward(int32_t rewardSpace, Reward* reward) {
  RETURN_IF_ERROR(rangeCheck(rewardSpace, 0, static_cast<int32_t>(rewardSpaces_.size()) - 1));
  reward->set_reward(0);
  return Status::OK;
}

}  // namespace compiler_gym::example_service

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmService.h"

#include <glog/logging.h>

#include <optional>

#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/envs/llvm/service/RewardSpaces.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/GrpcStatusMacros.h"

namespace compiler_gym::llvm_service {

using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;
namespace fs = boost::filesystem;

LlvmService::LlvmService(const fs::path& workingDirectory)
    : workingDirectory_(workingDirectory), benchmarkFactory_(workingDirectory), nextSessionId_(0) {}

Status LlvmService::GetSpaces(ServerContext* /* unused */, const GetSpacesRequest* /* unused */,
                              GetSpacesReply* reply) {
  VLOG(2) << "GetSpaces()";
  const auto actionSpaces = getLlvmActionSpaceList();
  *reply->mutable_action_space_list() = {actionSpaces.begin(), actionSpaces.end()};
  const auto observationSpaces = getLlvmObservationSpaceList();
  *reply->mutable_observation_space_list() = {observationSpaces.begin(), observationSpaces.end()};
  const auto rewardSpaces = getLlvmRewardSpaceList();
  *reply->mutable_reward_space_list() = {rewardSpaces.begin(), rewardSpaces.end()};

  return Status::OK;
}

Status LlvmService::StartEpisode(ServerContext* /* unused */, const StartEpisodeRequest* request,
                                 StartEpisodeReply* reply) {
  std::unique_ptr<Benchmark> benchmark;
  if (request->benchmark().size()) {
    RETURN_IF_ERROR(benchmarkFactory_.getBenchmark(request->benchmark(), &benchmark));
  } else {
    RETURN_IF_ERROR(benchmarkFactory_.getBenchmark(&benchmark));
  }

  reply->set_benchmark(benchmark->name());
  VLOG(1) << "StartEpisode(" << benchmark->name() << ")";

  LlvmActionSpace actionSpace;
  RETURN_IF_ERROR(util::intToEnum(request->action_space(), &actionSpace));

  // Set the eager observation space.
  std::optional<LlvmObservationSpace> eagerObservation = std::nullopt;
  if (request->use_eager_observation_space()) {
    const int32_t index = request->eager_observation_space();
    LlvmObservationSpace space;
    RETURN_IF_ERROR(util::intToEnum<LlvmObservationSpace>(index, &space));
    eagerObservation = space;
  }

  // Set the eager reward space.
  std::optional<LlvmRewardSpace> eagerReward = std::nullopt;
  if (request->use_eager_reward_space()) {
    const int32_t index = request->eager_reward_space();
    LlvmRewardSpace space;
    RETURN_IF_ERROR(util::intToEnum<LlvmRewardSpace>(index, &space));
    eagerReward = space;
  }

  // Construct the environment.
  reply->set_session_id(nextSessionId_);
  sessions_[nextSessionId_] = std::make_unique<LlvmEnvironment>(
      std::move(benchmark), actionSpace, eagerObservation, eagerReward, workingDirectory_);
  ++nextSessionId_;
  return Status::OK;
}

Status LlvmService::EndEpisode(grpc::ServerContext* /* unused */, const EndEpisodeRequest* request,
                               EndEpisodeReply* /* unused */) {
  // Note that unlike the other methods, no error is thrown if the requested
  // episode does not exist.
  if (sessions_.find(request->session_id()) == sessions_.end()) {
    return Status::OK;
  }

  const LlvmEnvironment* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));
  VLOG(1) << "Step " << environment->actionCount() << " EndEpisode("
          << environment->benchmark().name() << ")";

  return Status::OK;
}

Status LlvmService::TakeAction(ServerContext* /* unused */, const ActionRequest* request,
                               ActionReply* reply) {
  LlvmEnvironment* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));

  // Nothing was requested.
  if (!request->action_size()) {
    VLOG(2) << "Step " << environment->actionCount() << " TakeAction()";
    return Status::OK;
  }

  VLOG(2) << "Step " << environment->actionCount() << " TakeAction(" << request->action(0) << ")";
  RETURN_IF_ERROR(environment->takeAction(*request, reply));

  return Status::OK;
}

Status LlvmService::GetObservation(ServerContext* /* unused */, const ObservationRequest* request,
                                   Observation* reply) {
  LlvmEnvironment* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));

  const int32_t index = request->observation_space();
  VLOG(2) << "Step " << environment->actionCount() << " GetObservation(" << index << ")";

  LlvmObservationSpace space;
  RETURN_IF_ERROR(util::intToEnum(index, &space));
  RETURN_IF_ERROR(environment->getObservation(space, reply));

  return Status::OK;
}

Status LlvmService::GetReward(ServerContext* /* unused */, const RewardRequest* request,
                              Reward* reply) {
  LlvmEnvironment* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));

  const int32_t index = request->reward_space();
  VLOG(2) << "Step " << environment->actionCount() << " GetReward(" << index << ")";

  LlvmRewardSpace space;
  RETURN_IF_ERROR(util::intToEnum(index, &space));
  RETURN_IF_ERROR(environment->getReward(space, reply));

  return Status::OK;
}

Status LlvmService::AddBenchmark(ServerContext* /* unused */, const AddBenchmarkRequest* request,
                                 AddBenchmarkReply* reply) {
  VLOG(2) << "AddBenchmark()";
  for (int i = 0; i < request->benchmark_size(); ++i) {
    RETURN_IF_ERROR(addBenchmark(request->benchmark(i)));
  }

  return Status::OK;
}

Status LlvmService::addBenchmark(const ::compiler_gym::Benchmark& request) {
  const std::string& uri = request.uri();
  if (!uri.size()) {
    return Status(StatusCode::INVALID_ARGUMENT, "Benchmark must have a URI");
  }

  if (uri == "service://scan-site-data") {
    return benchmarkFactory_.scanSiteDataDirectory();
  }

  const auto& programFile = request.program();
  switch (programFile.data_case()) {
    case ::compiler_gym::File::DataCase::kContents:
      RETURN_IF_ERROR(benchmarkFactory_.addBitcode(
          uri, llvm::SmallString<0>(programFile.contents().begin(), programFile.contents().end())));
      break;
    case ::compiler_gym::File::DataCase::kUri:
      RETURN_IF_ERROR(benchmarkFactory_.addBitcodeUriAlias(uri, programFile.uri()));
      break;
    case ::compiler_gym::File::DataCase::DATA_NOT_SET:
      return Status(StatusCode::INVALID_ARGUMENT, "No program set");
  }

  return Status::OK;
}

Status LlvmService::GetBenchmarks(ServerContext* /* unused */,
                                  const GetBenchmarksRequest* /* unused */,
                                  GetBenchmarksReply* reply) {
  for (const auto& benchmark : benchmarkFactory_.getBenchmarkNames()) {
    reply->add_benchmark(benchmark);
  }

  return Status::OK;
}

Status LlvmService::session(uint64_t id, LlvmEnvironment** environment) {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return Status(StatusCode::INVALID_ARGUMENT, fmt::format("Session not found: {}", id));
  }

  *environment = it->second.get();
  return Status::OK;
}

Status LlvmService::session(uint64_t id, const LlvmEnvironment** environment) const {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return Status(StatusCode::INVALID_ARGUMENT, fmt::format("Session not found: {}", id));
  }

  *environment = it->second.get();
  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

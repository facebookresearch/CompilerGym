// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// An example implementation of the CompilerGymService interface.
#pragma once

#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::example_service {

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

  grpc::Status TakeAction(grpc::ServerContext* context, const ActionRequest* request,
                          ActionReply* reply) final override;

  grpc::Status GetObservation(grpc::ServerContext* context, const ObservationRequest* request,
                              Observation* reply) final override;

  grpc::Status GetReward(grpc::ServerContext* context, const RewardRequest* request,
                         Reward* reply) final override;

  grpc::Status GetBenchmarks(grpc::ServerContext* context, const GetBenchmarksRequest* request,
                             GetBenchmarksReply* reply) final override;

 private:
  [[nodiscard]] grpc::Status setObservation(int32_t observationSpace, Observation* observation);
  [[nodiscard]] grpc::Status setReward(int32_t rewardSpace, Reward* reward);

  const boost::filesystem::path workingDirectory_;
  bool episodeStarted_;
  std::vector<std::string> programNameList_;
  std::vector<std::string> actionSpace_;
  std::vector<ObservationSpace> observationSpaces_;
  std::vector<RewardSpace> rewardSpaces_;
  bool eagerObservation_;
  int32_t eagerObservationSpace_;
  bool eagerReward_;
  int32_t eagerRewardSpace_;
};

}  // namespace compiler_gym::example_service

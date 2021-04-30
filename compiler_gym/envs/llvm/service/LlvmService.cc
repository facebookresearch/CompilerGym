// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmService.h"

#include <glog/logging.h>

#include <optional>
#include <sstream>

#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/StrLenConstexpr.h"
#include "compiler_gym/util/Version.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Config/llvm-config.h"

namespace compiler_gym::llvm_service {

using grpc::ServerContext;
using grpc::Status;
using grpc::StatusCode;
namespace fs = boost::filesystem;

LlvmService::LlvmService(const fs::path& workingDirectory)
    : workingDirectory_(workingDirectory), benchmarkFactory_(workingDirectory), nextSessionId_(0) {}

Status LlvmService::GetVersion(ServerContext* /* unused */, const GetVersionRequest* /* unused */,
                               GetVersionReply* reply) {
  VLOG(2) << "GetSpaces()";
  reply->set_service_version(COMPILER_GYM_VERSION);
  std::stringstream ss;
  ss << LLVM_VERSION_STRING << " " << llvm::Triple::normalize(LLVM_DEFAULT_TARGET_TRIPLE);
  reply->set_compiler_version(ss.str());
  return Status::OK;
}

Status LlvmService::GetSpaces(ServerContext* /* unused */, const GetSpacesRequest* /* unused */,
                              GetSpacesReply* reply) {
  VLOG(2) << "GetSpaces()";
  const auto actionSpaces = getLlvmActionSpaceList();
  *reply->mutable_action_space_list() = {actionSpaces.begin(), actionSpaces.end()};
  const auto observationSpaces = getLlvmObservationSpaceList();
  *reply->mutable_observation_space_list() = {observationSpaces.begin(), observationSpaces.end()};

  return Status::OK;
}

Status LlvmService::StartSession(ServerContext* /* unused */, const StartSessionRequest* request,
                                 StartSessionReply* reply) {
  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  if (!request->benchmark().size()) {
    return Status(StatusCode::INVALID_ARGUMENT, "No benchmark URI set for StartSession()");
  }

  std::unique_ptr<Benchmark> benchmark;
  RETURN_IF_ERROR(benchmarkFactory_.getBenchmark(request->benchmark(), &benchmark));

  reply->set_benchmark(benchmark->name());
  VLOG(1) << "StartSession(" << benchmark->name() << "), [" << nextSessionId_ << "]";

  LlvmActionSpace actionSpace;
  RETURN_IF_ERROR(util::intToEnum(request->action_space(), &actionSpace));

  // Construct the environment.
  auto session =
      std::make_unique<LlvmSession>(std::move(benchmark), actionSpace, workingDirectory_);

  // Compute the initial observations.
  for (int i = 0; i < request->observation_space_size(); ++i) {
    LlvmObservationSpace observationSpace;
    RETURN_IF_ERROR(util::intToEnum(request->observation_space(i), &observationSpace));
    auto observation = reply->add_observation();
    RETURN_IF_ERROR(session->getObservation(observationSpace, observation));
  }

  reply->set_session_id(nextSessionId_);
  sessions_[nextSessionId_] = std::move(session);
  ++nextSessionId_;

  return Status::OK;
}

Status LlvmService::ForkSession(ServerContext* /* unused */, const ForkSessionRequest* request,
                                ForkSessionReply* reply) {
  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  LlvmSession* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));
  VLOG(1) << "ForkSession(" << request->session_id() << "), [" << nextSessionId_ << "]";

  // Construct the environment.
  reply->set_session_id(nextSessionId_);
  sessions_[nextSessionId_] =
      std::make_unique<LlvmSession>(environment->benchmark().clone(environment->workingDirectory()),
                                    environment->actionSpace(), environment->workingDirectory());

  ++nextSessionId_;

  return Status::OK;
}

Status LlvmService::EndSession(grpc::ServerContext* /* unused */, const EndSessionRequest* request,
                               EndSessionReply* reply) {
  const std::lock_guard<std::mutex> lock(sessionsMutex_);

  // Note that unlike the other methods, no error is thrown if the requested
  // session does not exist.
  if (sessions_.find(request->session_id()) != sessions_.end()) {
    const LlvmSession* environment;
    RETURN_IF_ERROR(session(request->session_id(), &environment));
    VLOG(1) << "Step " << environment->actionCount() << " EndSession("
            << environment->benchmark().name() << "), [" << request->session_id() << "]";

    sessions_.erase(request->session_id());
  }

  reply->set_remaining_sessions(sessions_.size());
  return Status::OK;
}

Status LlvmService::Step(ServerContext* /* unused */, const StepRequest* request,
                         StepReply* reply) {
  LlvmSession* environment;
  RETURN_IF_ERROR(session(request->session_id(), &environment));

  VLOG(2) << "Step " << environment->actionCount() << " Step()";
  return environment->step(*request, reply);
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

  const auto& programFile = request.program();
  switch (programFile.data_case()) {
    case ::compiler_gym::File::DataCase::kContents:
      return benchmarkFactory_.addBitcode(
          uri, llvm::SmallString<0>(programFile.contents().begin(), programFile.contents().end()));
    case ::compiler_gym::File::DataCase::kUri: {
      // Check that protocol of the benmchmark URI.
      if (programFile.uri().find("file:///") != 0) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      fmt::format("Invalid benchmark data URI. "
                                  "Only the file:/// protocol is supported: \"{}\"",
                                  programFile.uri()));
      }

      const fs::path path(programFile.uri().substr(util::strLen("file:///"), std::string::npos));
      return benchmarkFactory_.addBitcode(uri, path);
    }
    case ::compiler_gym::File::DataCase::DATA_NOT_SET:
      return Status(StatusCode::INVALID_ARGUMENT, "No program set");
  }

  return Status::OK;
}

Status LlvmService::session(uint64_t id, LlvmSession** environment) {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Session not found: {}", id));
  }

  *environment = it->second.get();
  return Status::OK;
}

Status LlvmService::session(uint64_t id, const LlvmSession** environment) const {
  auto it = sessions_.find(id);
  if (it == sessions_.end()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Session not found: {}", id));
  }

  *environment = it->second.get();
  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

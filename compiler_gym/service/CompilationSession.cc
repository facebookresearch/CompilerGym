// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/service/CompilationSession.h"

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym {

std::string CompilationSession::getCompilerVersion() const { return ""; }

Status CompilationSession::init(CompilationSession* other) {
  return Status(StatusCode::UNIMPLEMENTED, "CompilationSession::init() not implemented");
}

Status CompilationSession::endOfStep(bool actionHadNoEffect, bool& endOfEpisode,
                                     std::optional<ActionSpace>& newActionSpace) {
  return Status::OK;
}

Status CompilationSession::handleSessionParameter(const std::string& key, const std::string& value,
                                                  std::optional<std::string>& reply) {
  return Status::OK;
}

CompilationSession::CompilationSession(const boost::filesystem::path& workingDirectory)
    : workingDirectory_(workingDirectory) {}

}  // namespace compiler_gym

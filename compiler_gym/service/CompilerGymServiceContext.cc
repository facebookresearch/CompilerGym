// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/service/CompilerGymServiceContext.h"

using grpc::Status;

namespace compiler_gym {

CompilerGymServiceContext::CompilerGymServiceContext(
    const boost::filesystem::path& workingDirectory)
    : workingDirectory_(workingDirectory) {}

Status CompilerGymServiceContext::init() {
  VLOG(2) << "Initializing compiler service context";
  return Status::OK;
}

Status CompilerGymServiceContext::shutdown() {
  VLOG(2) << "Closing compiler service context";
  return Status::OK;
}

}  // namespace compiler_gym

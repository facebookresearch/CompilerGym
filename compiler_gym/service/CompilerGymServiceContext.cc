// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "compiler_gym/service/CompilerGymServiceContext.h"

using grpc::Status;

namespace compiler_gym {

CompilerGymServiceContext::CompilerGymServiceContext(
    const boost::filesystem::path& workingDirectory)
    : workingDirectory_(workingDirectory) {}

virtual Status CompilerGymServiceContext::init() { return Status::OK; }

virtual Status CompilerGymServiceContext::shutdown() { return Status::OK; }

}  // namespace compiler_gym

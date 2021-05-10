// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/service/core/Core.h"

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym {

grpc::Status CompilationSession::init(CompilationSession* other) {
  return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                      "copy initializer not supported for this compiler");
}

grpc::Status CompilationSession::endOfStep(bool* endOfEpisode) { return Status::OK; }

}  // namespace compiler_gym

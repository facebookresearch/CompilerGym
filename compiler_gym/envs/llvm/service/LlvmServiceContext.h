// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/service/CompilerGymServiceContext.h"

namespace compiler_gym::llvm_service {

class LlvmServiceContext final : public CompilerGymServiceContext {
 public:
  LlvmServiceContext(const boost::filesystem::path& workingDirectory);

  [[nodiscard]] virtual grpc::Status init() final override;

  [[nodiscard]] virtual grpc::Status shutdown() final override;

  BenchmarkFactory& benchmarkFactory() { return benchmarkFactory_; }

 private:
  BenchmarkFactory benchmarkFactory_;
};

}  // namespace compiler_gym::llvm_service

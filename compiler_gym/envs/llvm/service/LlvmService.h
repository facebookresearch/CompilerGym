// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <memory>
#include <mutex>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/envs/llvm/service/LlvmSession.h"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::llvm_service {

// RPC service for LLVM.
class LlvmService final : public CompilerGymService::Service {
 public:
  explicit LlvmService(const boost::filesystem::path& workingDirectory);

  // RPC endpoints.
  grpc::Status GetVersion(grpc::ServerContext* context, const GetVersionRequest* request,
                          GetVersionReply* reply) final override;

  grpc::Status GetSpaces(grpc::ServerContext* context, const GetSpacesRequest* request,
                         GetSpacesReply* reply) final override;

  grpc::Status StartSession(grpc::ServerContext* context, const StartSessionRequest* request,
                            StartSessionReply* reply) final override;

  grpc::Status ForkSession(grpc::ServerContext* context, const ForkSessionRequest* request,
                           ForkSessionReply* reply) final override;

  grpc::Status EndSession(grpc::ServerContext* context, const EndSessionRequest* request,
                          EndSessionReply* reply) final override;

  // NOTE: Step() is not thread safe. The underlying assumption is that each
  // LlvmSession is managed by a single thread, so race conditions between
  // operations that affect the same LlvmSession are not protected against.
  grpc::Status Step(grpc::ServerContext* context, const StepRequest* request,
                    StepReply* reply) final override;

  grpc::Status AddBenchmark(grpc::ServerContext* context, const AddBenchmarkRequest* request,
                            AddBenchmarkReply* reply) final override;

 protected:
  grpc::Status session(uint64_t id, LlvmSession** environment);
  grpc::Status session(uint64_t id, const LlvmSession** environment) const;

  grpc::Status addBenchmark(const ::compiler_gym::Benchmark& request);

 private:
  const boost::filesystem::path workingDirectory_;
  std::unordered_map<uint64_t, std::unique_ptr<LlvmSession>> sessions_;
  // Mutex used to ensure thread safety of creation and destruction of sessions.
  std::mutex sessionsMutex_;
  BenchmarkFactory benchmarkFactory_;
  uint64_t nextSessionId_;
};

}  // namespace compiler_gym::llvm_service

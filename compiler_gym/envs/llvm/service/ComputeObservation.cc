// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <fmt/format.h>
#include <glog/logging.h>

#include <boost/filesystem.hpp>
#include <iostream>
#include <magic_enum.hpp>

#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/envs/llvm/service/Observation.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

namespace fs = boost::filesystem;

using namespace compiler_gym;
using namespace compiler_gym::llvm_service;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  CHECK(argc == 3) << "Usage: compute_observation <observation-space> <bitcode-path>";

  const auto observationSpaceArg = magic_enum::enum_cast<LlvmObservationSpace>(argv[1]);
  CHECK(observationSpaceArg.has_value())
      << fmt::format("Invalid observation space name: {}", argv[1]);
  const LlvmObservationSpace observationSpace = observationSpaceArg.value();

  fs::path workingDirectory{"."};

  compiler_gym::Benchmark benchmarkMessage;
  benchmarkMessage.set_uri("user");
  benchmarkMessage.mutable_program()->set_uri(fmt::format("file:///{}", argv[2]));

  auto& benchmarkFactory = BenchmarkFactory::getSingleton(workingDirectory);
  std::unique_ptr<::llvm_service::Benchmark> benchmark;
  {
    const auto status = benchmarkFactory.getBenchmark(benchmarkMessage, &benchmark);
    CHECK(status.ok()) << "Failed to compute observation: " << status.error_message();
  }

  Event observation;
  {
    const auto status = setObservation(observationSpace, workingDirectory, *benchmark, observation);
    CHECK(status.ok()) << "Failed to compute observation: " << status.error_message();
  }

  std::cout << observation.DebugString() << std::endl;

  return 0;
}

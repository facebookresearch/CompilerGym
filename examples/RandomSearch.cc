// A random search for LLVM codesize using the C++ API.
//
// While not intended for the majority of users, it is entirely straightforward
// to skip the Python frontend and interact with the C++ API directly. This file
// demonstrates a simple parallelized random search implemented for the LLVM
// compiler service.
//
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <stdlib.h>
#include <time.h>

#include <boost/filesystem.hpp>
#include <iostream>
#include <limits>
#include <magic_enum.hpp>
#include <thread>
#include <vector>

#include "compiler_gym/envs/llvm/service/LlvmService.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/util/GrpcStatusMacros.h"

DEFINE_string(benchmark, "benchmark://cbench-v1/crc32", "The benchmark to use.");
DEFINE_int32(step_count, 100, "The number of steps to run for each random search");
DEFINE_int32(nproc, std::max(1u, std::thread::hardware_concurrency()),
             "The number of parallel search threads to use");

namespace fs = boost::filesystem;

namespace compiler_gym {

using grpc::Status;
using llvm_service::LlvmAction;
using llvm_service::LlvmObservationSpace;
using llvm_service::LlvmService;

// A wrapper around an LLVM service. Here, we call the RPC enpoints directly
// on the service, we do not use RPC. This means that we do not get the
// reliability benefits of running the compiler service in a separate process,
// but we also do not pay the performance overhead.
template <LlvmObservationSpace observationSpace>
class Environment {
 public:
  Environment(const fs::path& workingDir, const std::string& benchmark)
      : service_(workingDir), benchmark_(benchmark) {}

  // Reset the environment and compute the initial observation.
  [[nodiscard]] Status reset(Observation* observation) {
    if (inEpisode_) {
      RETURN_IF_ERROR(close());
    }

    StartSessionRequest startRequest;
    StartSessionReply startReply;
    startRequest.set_benchmark(benchmark_);
    RETURN_IF_ERROR(service_.StartSession(nullptr, &startRequest, &startReply));
    sessionId_ = startReply.session_id();

    StepRequest stepRequest;
    StepReply stepReply;
    stepRequest.set_session_id(sessionId_);
    stepRequest.add_observation_space(static_cast<int>(observationSpace));
    RETURN_IF_ERROR(service_.Step(nullptr, &stepRequest, &stepReply));
    CHECK(stepReply.observation_size() == 1);
    *observation = stepReply.observation(0);

    inEpisode_ = true;

    return Status::OK;
  }

  // End the current session.
  [[nodiscard]] Status close() {
    EndSessionRequest endRequest;
    EndSessionReply endReply;
    endRequest.set_session_id(sessionId_);
    inEpisode_ = false;
    return service_.EndSession(nullptr, &endRequest, &endReply);
  }

  // Apply the given action and compute an observation.
  [[nodiscard]] Status step(LlvmAction action, Observation* observation) {
    StepRequest request;
    StepReply reply;

    request.set_session_id(sessionId_);
    request.add_action(static_cast<int>(action));
    request.add_observation_space(static_cast<int>(observationSpace));
    RETURN_IF_ERROR(service_.Step(nullptr, &request, &reply));
    CHECK(reply.observation_size() == 1);
    *observation = reply.observation(0);
    return Status::OK;
  }

 private:
  LlvmService service_;
  const std::string benchmark_;
  bool inEpisode_;
  int64_t sessionId_;
};

Status runSearch(const fs::path& workingDir, std::vector<int>* bestActions, int64_t* bestCost) {
  Environment<LlvmObservationSpace::IR_INSTRUCTION_COUNT> environment(workingDir, FLAGS_benchmark);

  // Reset the environment.
  Observation init;
  RETURN_IF_ERROR(environment.reset(&init));
  *bestCost = init.scalar_int64();

  // Run a bunch of actions randomly.
  srand(time(NULL));
  std::vector<int> actions;
  for (int i = 0; i < FLAGS_step_count; ++i) {
    int action = rand() % magic_enum::enum_count<LlvmAction>();
    actions.push_back(action);

    Observation obs;
    RETURN_IF_ERROR(environment.step(static_cast<LlvmAction>(action), &obs));
    int64_t cost = obs.scalar_int64();

    if (cost < *bestCost) {
      *bestCost = cost;
      *bestActions = actions;
    }
    VLOG(3) << "Step " << action << " " << cost << " " << *bestCost;
  }

  RETURN_IF_ERROR(environment.close());

  return Status::OK;
}

void runThread(std::vector<int>* bestActions, int64_t* bestCost) {
  const fs::path workingDir = fs::unique_path();
  fs::create_directories(workingDir);
  if (!runSearch(workingDir, bestActions, bestCost).ok()) {
    LOG(ERROR) << "Search failed";
  }
  fs::remove_all(workingDir);
}

// Run `numThreads` random searches concurrently.
Status runRandomSearches(unsigned numThreads) {
  std::cout << "Starting " << numThreads << " random search threads for benchmark "
            << FLAGS_benchmark << std::endl;
  std::vector<std::thread> threads;
  std::vector<std::vector<int>> actions(numThreads);
  std::vector<int64_t> costs(numThreads, INT_MAX);

  for (unsigned i = 0; i < numThreads; ++i) {
    threads.push_back(std::thread(runThread, &actions[i], &costs[i]));
  }

  for (auto& thread : threads) {
    thread.join();
  }

  int64_t bestCost = costs[0];
  unsigned bestThread = 0;
  for (unsigned i = 0; i < costs.size(); ++i) {
    if (costs[i] < bestCost) {
      bestCost = costs[i];
      bestThread = i;
    }
  }

  std::cout << "Lowest cost achieved: " << bestCost << std::endl;
  std::cout << "Actions: ";
  for (auto action : actions[bestThread]) {
    std::cout << action << " ";
  }
  std::cout << std::endl;

  return Status::OK;
}

}  // namespace compiler_gym

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/false);
  google::InitGoogleLogging(argv[0]);
  CHECK(compiler_gym::runRandomSearches(FLAGS_nproc).ok());
}

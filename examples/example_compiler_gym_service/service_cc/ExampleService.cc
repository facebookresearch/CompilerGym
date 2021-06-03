// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Run the example service on a local port.
#include <fmt/format.h>

#include "compiler_gym/service/CompilationSession.h"
#include "compiler_gym/service/runtime/Runtime.h"
#include "compiler_gym/util/Unreachable.h"

const char* usage = R"(Example CompilerGym service)";

using namespace compiler_gym;
using grpc::Status;
using grpc::StatusCode;

namespace {

template <typename T>
[[nodiscard]] inline Status rangeCheck(const T& value, const T& minValue, const T& maxValue) {
  if (value < minValue || value > maxValue) {
    return Status(StatusCode::INVALID_ARGUMENT, "Out-of-range");
  }
  return Status::OK;
}

class ExampleCompilationSession final : public CompilationSession {
 public:
  using CompilationSession::CompilationSession;

  std::string getCompilerVersion() const final override { return "1.0.0"; }

  std::vector<ActionSpace> getActionSpaces() const final override {
    ActionSpace space;
    space.set_name("default");
    space.add_action("a");
    space.add_action("b");
    space.add_action("c");

    return {space};
  }

  std::vector<ObservationSpace> getObservationSpaces() const override {
    ObservationSpace ir;
    ir.set_name("ir");
    ScalarRange irSizeRange;
    irSizeRange.mutable_min()->set_value(0);
    *ir.mutable_string_size_range() = irSizeRange;
    ir.set_deterministic(true);
    ir.set_platform_dependent(false);

    ObservationSpace features;
    features.set_name("features");
    for (int i = 0; i < 3; ++i) {
      ScalarRange* featureSizeRange = features.mutable_int64_range_list()->add_range();
      featureSizeRange->mutable_min()->set_value(-100);
      featureSizeRange->mutable_max()->set_value(100);
    }

    ObservationSpace runtime;
    runtime.set_name("runtime");
    ScalarRange runtimeRange;
    runtimeRange.mutable_min()->set_value(0);
    *runtime.mutable_scalar_double_range() = runtimeRange;
    runtime.set_deterministic(false);
    runtime.set_platform_dependent(true);

    return {ir, features, runtime};
  }

  [[nodiscard]] grpc::Status init(const ActionSpace& actionSpace,
                                  const compiler_gym::Benchmark& benchmark) final override {
    VLOG(1) << "Starting a compilation session for " << benchmark.uri();
    return Status::OK;
  }

  [[nodiscard]] grpc::Status init(CompilationSession* other) final override {
    VLOG(1) << "Forking the compilation session";
    return Status::OK;
  }

  [[nodiscard]] grpc::Status applyAction(const Action& action, bool& endOfEpisode,
                                         std::optional<ActionSpace>& newActionSpace,
                                         bool& actionHadNoEffect) final override {
    LOG(INFO) << "Applying action " << action.action();

    if (action.action() < 0 || action.action() > getActionSpaces()[0].action_size()) {
      return Status(StatusCode::INVALID_ARGUMENT, "Out-of-range");
    }

    return Status::OK;
  }

  [[nodiscard]] grpc::Status computeObservation(const ObservationSpace& observationSpace,
                                                Observation& observation) final override {
    std::cerr << "COMPUTING OBSERVATION" << std::endl;
    LOG(INFO) << "Computing observation " << observationSpace.name();
    std::cerr << "CP2" << std::endl;

    if (observationSpace.name() == "ir") {
      std::cerr << "IR" << std::endl;
      observation.set_string_value("Hello, world!");
    } else if (observationSpace.name() == "features") {
      std::cerr << "IR" << std::endl;
      for (int i = 0; i < 3; ++i) {
        observation.mutable_int64_list()->add_value(0);
      }
    } else if (observationSpace.name() == "runtime") {
      std::cerr << "IR" << std::endl;
      observation.set_scalar_double(0);
    } else {
      UNREACHABLE(fmt::format("Unhandled observation space: {}", observationSpace.name()));
    }

    return Status::OK;

    std::cerr << "DONE" << std::endl;
  }
};

}  // namespace

int main(int argc, char** argv) {
  runtime::createAndRunCompilerGymService<ExampleCompilationSession>(argc, argv, usage);
}

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Run the example service on a local port.
#include "compiler_gym/service/core/Core.h"
#include "compiler_gym/service/core/Run.h"

const char* usage = R"(Example CompilerGym service)";

using namespace compiler_gym;
using grpc::Status;
using grpc::StatusCode;

namespace fs = boost::filesystem;

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

  std::string getCompilerVersion() const override { return "1.0.0"; }

  std::vector<ActionSpace> getActionSpaces() const override {
    ActionSpace space;
    space.set_name("default");
    space.add_action("a");
    space.add_action("b");
    space.add_action("c");

    return {space};
  }

  ActionSpace getActionSpace() const override { return actionSpace_; };

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

  [[nodiscard]] grpc::Status init(size_t actionSpaceIndex,
                                  const compiler_gym::Benchmark& benchmark) override {
    VLOG(1) << "Started a compilation session for " << benchmark.uri();
    const auto actionSpaces = getActionSpaces();
    RETURN_IF_ERROR(rangeCheck(actionSpaceIndex, 0ul, actionSpaces.size() - 1));
    actionSpace_ = actionSpaces[actionSpaceIndex];
    return Status::OK;
  }

  [[nodiscard]] grpc::Status init(CompilationSession* other) override {
    VLOG(1) << "Forked compilation session";
    actionSpace_ = static_cast<ExampleCompilationSession*>(other)->actionSpace_;
    return Status::OK;
  }

  [[nodiscard]] grpc::Status applyAction(size_t actionIndex, bool* endOfEpisode,
                                         bool* actionSpaceChanged,
                                         bool* actionHadNoEffect) override {
    RETURN_IF_ERROR(
        rangeCheck(actionIndex, 0ul, static_cast<size_t>(actionSpace_.action_size() - 1)));

    return Status::OK;
  }

  [[nodiscard]] grpc::Status setObservation(size_t observationSpaceIndex,
                                            Observation* observation) override {
    const auto observationSpaces = getObservationSpaces();
    RETURN_IF_ERROR(rangeCheck(observationSpaceIndex, 0ul, observationSpaces.size() - 1));
    switch (observationSpaceIndex) {
      case 0:  // IR
        observation->set_string_value("Hello, world!");
        break;
      case 1:  // Features
        for (int i = 0; i < 3; ++i) {
          observation->mutable_int64_list()->add_value(0);
        }
        break;
      case 2:  // Runtime
        observation->set_scalar_double(0);
      default:
        break;
    }
    return Status::OK;
  }

 private:
  ActionSpace actionSpace_;
};

}  // namespace

int main(int argc, char** argv) {
  createAndRunCompilerGymService<ExampleCompilationSession>(argc, argv, usage);
}

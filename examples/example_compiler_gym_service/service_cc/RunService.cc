// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Run the example service on a local port.
#include "compiler_gym/service/core/Core.h"
#include "compiler_gym/service/core/Run.h"
#include "examples/example_compiler_gym_service/service_cc/ExampleService.h"

const char* usage = R"(Example CompilerGym service)";

using compiler_gym::createAndRunCompilerGymService;
using compiler_gym::example_service::ExampleCompilationSession;

int main(int argc, char** argv) {
  createAndRunCompilerGymService<ExampleCompilationSession>(&argc, &argv, usage);
}

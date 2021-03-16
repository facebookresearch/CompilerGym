// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Run the example service on a local port.
#include "compiler_gym/util/RunService.h"

#include "examples/example_compiler_gym_service/service_cc/ExampleService.h"

const char* usage = R"(LLVM CompilerGym service)";

using namespace compiler_gym::util;
using namespace compiler_gym::example_service;

int main(int argc, char** argv) {
  // Call the utility runService() function to launch the service. This function
  // never returns.
  return runService<ExampleService>(&argc, &argv, usage);
}

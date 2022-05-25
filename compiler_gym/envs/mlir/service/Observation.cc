// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/mlir/service/Observation.h"

#include <cpuinfo.h>
#include <glog/logging.h>

#include <iomanip>
#include <sstream>
#include <string>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/mlir/service/Benchmark.h"
#include "compiler_gym/envs/mlir/service/ObservationSpaces.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "nlohmann/json.hpp"

namespace fs = boost::filesystem;

namespace compiler_gym::mlir_service {

using grpc::Status;
using grpc::StatusCode;
using nlohmann::json;

Status setObservation(MlirObservationSpace space, const fs::path& workingDirectory,
                      Benchmark& benchmark, Event& reply) {
  switch (space) {
    case MlirObservationSpace::RUNTIME: {
      return benchmark.computeRuntime(reply);
    }
    case MlirObservationSpace::IS_RUNNABLE: {
      reply.set_boolean_value(benchmark.isRunnable());
      break;
    }
  }

  return Status::OK;
}

}  // namespace compiler_gym::mlir_service

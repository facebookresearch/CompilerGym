

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/mlir/service/Benchmark.h"
#include "compiler_gym/envs/mlir/service/ObservationSpaces.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::mlir_service {

/**
 * Compute an observation using the given space.
 *
 * @param space The observation space to compute.
 * @param workingDirectory A scratch directory.
 * @param benchmark The benchmark to compute the observation on.
 * @param reply The observation to set.
 * @return `OK` on success.
 */
grpc::Status setObservation(MlirObservationSpace space,
                            const boost::filesystem::path& workingDirectory, Benchmark& benchmark,
                            Event& reply);

}  // namespace compiler_gym::mlir_service

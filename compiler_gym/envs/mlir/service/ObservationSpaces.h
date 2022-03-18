// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <vector>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::mlir_service {

/**
 * The available observation spaces for MLIR.
 *
 * \note Housekeeping rules - to add a new observation space:
 *   1. Add a new entry to this MlirObservationSpace enum.
 *   2. Add a new switch case to getMlirObservationSpaceList() to return the
 *      ObserverationSpace.
 *   3. Add a new switch case to MlirSession::getObservation() to compute
 *      the actual observation.
 *   4. Run `bazel test //compiler_gym/...` and update the newly failing tests.
 */
enum class MlirObservationSpace {
  /** Return 1 if the benchmark is runnable, else 0.
   */
  IS_RUNNABLE,
  /** The runtime of the compiled program.
   *
   * Returns a list of runtime measurements in microseconds. This is not
   * available to all benchmarks. When not available, a list of zeros are returned.
   */
  RUNTIME,
};

/** Return the list of available observation spaces. */
std::vector<ObservationSpace> getMlirObservationSpaceList();

}  // namespace compiler_gym::mlir_service

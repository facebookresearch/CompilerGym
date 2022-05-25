// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::mlir_service {

/**
 * The available action spaces for MLIR.
 *
 * \note Implementation housekeeping rules - to add a new action space:
 *   1. Add a new entry to this MlirActionSpace enum.
 *   2. Add a new switch case to getMlirActionSpaceList() to return the
 *      ActionSpace.
 *   3. Add a new switch case to MlirSession::step() to compute
 *      the actual action.
 *   4. Run `bazel test //compiler_gym/...` and update the newly failing tests.
 */
enum class MlirActionSpace { MATRIX_MULTIPLICATION };

/**
 * Get the list of MLIR action spaces.
 *
 * @return A list of ActionSpace instances.
 */
std::vector<ActionSpace> getMlirActionSpaceList();

}  // namespace compiler_gym::mlir_service

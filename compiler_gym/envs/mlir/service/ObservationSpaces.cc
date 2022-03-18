// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/mlir/service/ObservationSpaces.h"

#include <magic_enum.hpp>

#include "compiler_gym/util/EnumUtil.h"

namespace compiler_gym::mlir_service {

std::vector<ObservationSpace> getMlirObservationSpaceList() {
  std::vector<ObservationSpace> spaces;
  spaces.reserve(magic_enum::enum_count<MlirObservationSpace>());
  for (const auto& value : magic_enum::enum_values<MlirObservationSpace>()) {
    ObservationSpace space;
    space.set_name(util::enumNameToPascalCase<MlirObservationSpace>(value));
    switch (value) {
      case MlirObservationSpace::RUNTIME: {
        space.mutable_space()->mutable_double_sequence()->mutable_length_range()->set_min(0);
        space.mutable_space()->mutable_double_sequence()->mutable_scalar_range()->set_min(0);
        space.set_deterministic(false);
        space.set_platform_dependent(true);
        break;
      }
      case MlirObservationSpace::IS_RUNNABLE: {
        space.mutable_space()->mutable_boolean_value();
        space.set_deterministic(true);
        space.set_platform_dependent(true);
        space.mutable_default_observation()->set_boolean_value(true);
        break;
      }
    }
    spaces.push_back(space);
  }
  return spaces;
}

}  // namespace compiler_gym::mlir_service

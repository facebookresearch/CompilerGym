// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/RewardSpaces.h"

#include <magic_enum.hpp>

#include "compiler_gym/util/EnumUtil.h"

namespace compiler_gym::llvm_service {

std::vector<RewardSpace> getLlvmRewardSpaceList() {
  std::vector<RewardSpace> spaces;
  spaces.reserve(magic_enum::enum_count<LlvmRewardSpace>());
  for (const auto& value : magic_enum::enum_values<LlvmRewardSpace>()) {
    RewardSpace space;
    space.set_name(util::enumNameToPascalCase<LlvmRewardSpace>(value));
    switch (value) {
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT:
        space.set_deterministic(true);
        break;
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_O3:
        space.set_deterministic(true);
        space.set_has_success_threshold(true);
        space.set_success_threshold(1.0);
        break;
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_Oz:
        space.set_deterministic(true);
        space.set_has_success_threshold(true);
        space.set_success_threshold(1.0);
        break;
      case LlvmRewardSpace::OBJECT_TEXT_SIZE_BYTES:
        space.set_deterministic(true);
        space.set_platform_dependent(true);
        break;
      case LlvmRewardSpace::OBJECT_TEXT_SIZE_O3:
        space.set_deterministic(true);
        space.set_has_success_threshold(true);
        space.set_success_threshold(1.0);
        space.set_platform_dependent(true);
        break;
      case LlvmRewardSpace::OBJECT_TEXT_SIZE_Oz:
        space.set_deterministic(true);
        space.set_has_success_threshold(true);
        space.set_success_threshold(1.0);
        space.set_platform_dependent(true);
        break;
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
      case LlvmRewardSpace::TEXT_SIZE_BYTES:
        space.set_deterministic(true);
        space.set_platform_dependent(true);
        break;
      case LlvmRewardSpace::TEXT_SIZE_O3:
        space.set_deterministic(true);
        space.set_has_success_threshold(true);
        space.set_success_threshold(1.0);
        space.set_platform_dependent(true);
        break;
      case LlvmRewardSpace::TEXT_SIZE_Oz:
        space.set_deterministic(true);
        space.set_has_success_threshold(true);
        space.set_success_threshold(1.0);
        space.set_platform_dependent(true);
        break;
#endif
    }
    spaces.push_back(space);
  }
  return spaces;
}

}  // namespace compiler_gym::llvm_service

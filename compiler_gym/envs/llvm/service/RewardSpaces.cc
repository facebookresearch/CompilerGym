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
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_O3:
      case LlvmRewardSpace::IR_INSTRUCTION_COUNT_Oz:
      case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES:
      case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES_O3:
      case LlvmRewardSpace::NATIVE_TEXT_SIZE_BYTES_Oz:
        // These reward spaces have no bounds.
        break;
    }
    spaces.push_back(space);
  }
  return spaces;
}

}  // namespace compiler_gym::llvm_service

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/ActionSpace.h"

#include <fmt/format.h>

#include <magic_enum.hpp>

#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/Unreachable.h"

namespace compiler_gym::llvm_service {

std::vector<ActionSpace> getLlvmActionSpaceList() {
  std::vector<ActionSpace> spaces;
  spaces.reserve(magic_enum::enum_count<LlvmActionSpace>());
  for (const auto& enumValue : magic_enum::enum_values<LlvmActionSpace>()) {
    ActionSpace actionSpace;
    actionSpace.set_name(util::enumNameToPascalCase<LlvmActionSpace>(enumValue));
    Space& space = *actionSpace.mutable_space();
    switch (enumValue) {
      case LlvmActionSpace::PASSES_ALL: {
        CommandlineSpace flagValue;
        for (const auto& enumValue : magic_enum::enum_values<LlvmAction>()) {
          flagValue.add_name(util::enumNameToCommandlineFlag<LlvmAction>(enumValue));
        }
        space.mutable_any_value()->PackFrom(flagValue);
      } break;
      default:
        UNREACHABLE(fmt::format("Unknown LLVM action space {}",
                                util::enumNameToPascalCase<LlvmActionSpace>(enumValue)));
    }

    spaces.push_back(actionSpace);
  }
  return spaces;
}

}  // namespace compiler_gym::llvm_service

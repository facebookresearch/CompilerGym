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
  for (const auto& value : magic_enum::enum_values<LlvmActionSpace>()) {
    ActionSpace space;
    space.set_name(util::enumNameToPascalCase<LlvmActionSpace>(value));
    switch (value) {
      case LlvmActionSpace::PASSES_ALL: {
        ChoiceSpace* flagChoice = space.add_choice();
        flagChoice->set_name("flag");

        NamedDiscreteSpace* flagChoiceSpace = flagChoice->mutable_named_discrete_space();
        flagChoiceSpace->set_is_commandline(true);
        for (const auto& value : magic_enum::enum_values<LlvmAction>()) {
          flagChoiceSpace->add_value(util::enumNameToCommandlineFlag<LlvmAction>(value));
        }
      } break;
      default:
        UNREACHABLE(fmt::format("Unknown LLVM action space {}",
                                util::enumNameToPascalCase<LlvmActionSpace>(value)));
    }

    spaces.push_back(space);
  }
  return spaces;
}

}  // namespace compiler_gym::llvm_service

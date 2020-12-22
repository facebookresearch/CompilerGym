// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"

#include <magic_enum.hpp>

#include "compiler_gym/util/EnumUtil.h"

namespace compiler_gym::llvm_service {

std::vector<ObservationSpace> getLlvmObservationSpaceList() {
  std::vector<ObservationSpace> spaces;
  spaces.reserve(magic_enum::enum_count<LlvmObservationSpace>());
  for (const auto& value : magic_enum::enum_values<LlvmObservationSpace>()) {
    ObservationSpace space;
    space.set_name(util::enumNameToPascalCase<LlvmObservationSpace>(value));
    switch (value) {
      case LlvmObservationSpace::IR: {
        ScalarRange irSize;
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        break;
      }
      case LlvmObservationSpace::BITCODE_FILE: {
        ScalarRange pathLength;
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        space.mutable_string_size_range()->mutable_max()->set_value(4096);
        break;
      }
      case LlvmObservationSpace::AUTOPHASE: {
        ScalarRange featureSize;
        featureSize.mutable_min()->set_value(0);
        std::vector<ScalarRange> featureSizes;
        for (int i = 0; i < /* autophase feature vector dim: */ 56; ++i) {
          featureSizes.push_back(featureSize);
        }
        *space.mutable_int64_range_list()->mutable_range() = {featureSizes.begin(),
                                                              featureSizes.end()};
        break;
      }
      case LlvmObservationSpace::PROGRAML: {
        // ProGraML serializes the graph to JSON.
        ScalarRange encodedSize;
        encodedSize.mutable_min()->set_value(0);
        space.set_opaque_data_format("json://networkx/MultiDiGraph");
        *space.mutable_string_size_range() = encodedSize;
        break;
      }
      case LlvmObservationSpace::CPU_INFO: {
        // Hardware info is returned as a JSON
        ScalarRange encodedSize;
        encodedSize.mutable_min()->set_value(0);
        space.set_opaque_data_format("json://");
        *space.mutable_string_size_range() = encodedSize;
        break;
      }
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ:
      case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES:
      case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES_O0:
      case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES_O3:
      case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES_OZ: {
        auto featureSize = space.mutable_int64_range_list()->add_range();
        featureSize->mutable_min()->set_value(0);
        break;
      }
    }
    spaces.push_back(space);
  }
  return spaces;
}

}  // namespace compiler_gym::llvm_service

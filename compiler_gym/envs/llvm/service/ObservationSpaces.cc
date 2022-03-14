// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"

#include <glog/logging.h>

#include <magic_enum.hpp>

#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/third_party/llvm/InstCount.h"
#include "compiler_gym/util/EnumUtil.h"
#include "nlohmann/json.hpp"
#include "programl/graph/format/node_link_graph.h"
#include "programl/proto/program_graph.pb.h"

using nlohmann::json;

namespace compiler_gym::llvm_service {

// The number of features in the Autophase feature vector.
static constexpr size_t kAutophaseFeatureDim = 56;

// The number of features in the IR2Vec feature vector.
static constexpr size_t kIR2VecFeatureDim = 300;

// 4096 is the maximum path length for most filesystems.
static constexpr size_t kMaximumPathLength = 4096;

std::vector<ObservationSpace> getLlvmObservationSpaceList() {
  std::vector<ObservationSpace> spaces;
  spaces.reserve(magic_enum::enum_count<LlvmObservationSpace>());
  for (const auto& value : magic_enum::enum_values<LlvmObservationSpace>()) {
    ObservationSpace space;
    space.set_name(util::enumNameToPascalCase<LlvmObservationSpace>(value));
    switch (value) {
      case LlvmObservationSpace::IR: {
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::IR_SHA1: {
        space.mutable_string_size_range()->mutable_min()->set_value(40);
        space.mutable_string_size_range()->mutable_max()->set_value(40);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::BITCODE: {
        space.mutable_binary_size_range()->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::BITCODE_FILE: {
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        // 4096 is the maximum path length for most filesystems.
        space.mutable_string_size_range()->mutable_max()->set_value(kMaximumPathLength);
        // A random file path is generated, so the returned value is not
        // deterministic.
        space.set_deterministic(false);
        space.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::INST_COUNT: {
        ScalarRange featureSize;
        featureSize.mutable_min()->set_value(0);
        std::vector<ScalarRange> featureSizes(kInstCountFeatureDimensionality, featureSize);
        *space.mutable_int64_range_list()->mutable_range() = {featureSizes.begin(),
                                                              featureSizes.end()};
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        std::vector<int64_t> defaultValue(kInstCountFeatureDimensionality, 0);
        *space.mutable_default_value()->mutable_int64_list()->mutable_value() = {
            defaultValue.begin(), defaultValue.end()};
        break;
      }
      case LlvmObservationSpace::AUTOPHASE: {
        ScalarRange featureSize;
        featureSize.mutable_min()->set_value(0);
        std::vector<ScalarRange> featureSizes;
        featureSizes.reserve(kAutophaseFeatureDim);
        for (size_t i = 0; i < kAutophaseFeatureDim; ++i) {
          featureSizes.push_back(featureSize);
        }
        *space.mutable_int64_range_list()->mutable_range() = {featureSizes.begin(),
                                                              featureSizes.end()};
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        std::vector<int64_t> defaultValue(kAutophaseFeatureDim, 0);
        *space.mutable_default_value()->mutable_int64_list()->mutable_value() = {
            defaultValue.begin(), defaultValue.end()};
        break;
      }
      case LlvmObservationSpace::IR2VEC_FLOW_AWARE: {
        ScalarRange featureSize;
        std::vector<ScalarRange> featureSizes;
        featureSizes.reserve(kIR2VecFeatureDim);
        for (size_t i = 0; i < kIR2VecFeatureDim; ++i) {
          featureSizes.push_back(featureSize);
        }
        *space.mutable_double_range_list()->mutable_range() = {featureSizes.begin(),
                                                               featureSizes.end()};
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        std::vector<double> defaultValue(kIR2VecFeatureDim, 0.0);
        *space.mutable_default_value()->mutable_double_list()->mutable_value() = {
            defaultValue.begin(), defaultValue.end()};
        break;
      }
      case LlvmObservationSpace::IR2VEC_SYMBOLIC: {
        ScalarRange featureSize;
        std::vector<ScalarRange> featureSizes;
        featureSizes.reserve(kIR2VecFeatureDim);
        for (size_t i = 0; i < kIR2VecFeatureDim; ++i) {
          featureSizes.push_back(featureSize);
        }
        *space.mutable_double_range_list()->mutable_range() = {featureSizes.begin(),
                                                               featureSizes.end()};
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        std::vector<double> defaultValue(kIR2VecFeatureDim, 0.0);
        *space.mutable_default_value()->mutable_double_list()->mutable_value() = {
            defaultValue.begin(), defaultValue.end()};
        break;
      }
      case LlvmObservationSpace::IR2VEC_FUNCTION_LEVEL_FLOW_AWARE: {
        space.set_opaque_data_format("json://");
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        std::vector<double> defaultEmbs;
        for (double i = 0; i < kIR2VecFeatureDim; i++) {
          defaultEmbs.push_back(i);
        }
        json vectorJson = defaultEmbs;
        json FunctionKey;
        json embeddings;
        FunctionKey["default"] = vectorJson;
        embeddings["embeddings"] = FunctionKey;
        *space.mutable_default_value()->mutable_string_value() = embeddings.dump();
        break;
      }
      case LlvmObservationSpace::IR2VEC_FUNCTION_LEVEL_SYMBOLIC: {
        space.set_opaque_data_format("json://");
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        std::vector<double> defaultEmbs;
        for (double i = 0; i < kIR2VecFeatureDim; i++) {
          defaultEmbs.push_back(i);
        }
        json vectorJson = defaultEmbs;
        json FunctionKey;
        json embeddings;
        FunctionKey["default"] = vectorJson;
        embeddings["embeddings"] = FunctionKey;
        *space.mutable_default_value()->mutable_string_value() = embeddings.dump();
        break;
      }
      case LlvmObservationSpace::PROGRAML: {
        // ProGraML serializes the graph to JSON.
        space.set_opaque_data_format("json://networkx/MultiDiGraph");
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        programl::ProgramGraph graph;
        json nodeLinkGraph;
        CHECK(programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph).ok())
            << "Failed to serialize default ProGraML graph";
        *space.mutable_default_value()->mutable_string_value() = nodeLinkGraph.dump();
        break;
      }
      case LlvmObservationSpace::PROGRAML_JSON: {
        // ProGraML serializes the graph to JSON.
        space.set_opaque_data_format("json://");
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        programl::ProgramGraph graph;
        json nodeLinkGraph;
        CHECK(programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph).ok())
            << "Failed to serialize default ProGraML graph";
        *space.mutable_default_value()->mutable_string_value() = nodeLinkGraph.dump();
        break;
      }
      case LlvmObservationSpace::CPU_INFO: {
        // Hardware info is returned as a JSON
        space.set_opaque_data_format("json://");
        space.mutable_string_size_range()->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(true);
        *space.mutable_default_value()->mutable_string_value() = "{}";
        break;
      }
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ: {
        auto featureSize = space.mutable_scalar_int64_range();
        featureSize->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(false);
        space.mutable_default_value()->set_scalar_int64(0);
        break;
      }
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_BYTES:
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_O0:
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_O3:
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_OZ: {
        auto featureSize = space.mutable_scalar_int64_range();
        featureSize->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(true);
        space.mutable_default_value()->set_scalar_int64(0);
        break;
      }
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
      case LlvmObservationSpace::TEXT_SIZE_BYTES:
      case LlvmObservationSpace::TEXT_SIZE_O0:
      case LlvmObservationSpace::TEXT_SIZE_O3:
      case LlvmObservationSpace::TEXT_SIZE_OZ: {
        auto featureSize = space.mutable_scalar_int64_range();
        featureSize->mutable_min()->set_value(0);
        space.set_deterministic(true);
        space.set_platform_dependent(true);
        space.mutable_default_value()->set_scalar_int64(0);
        break;
      }
#endif
      case LlvmObservationSpace::RUNTIME: {
        space.mutable_double_sequence()->mutable_length_range()->mutable_min()->set_value(0);
        space.mutable_double_sequence()->mutable_scalar_range()->mutable_min()->set_value(0);
        space.set_deterministic(false);
        space.set_platform_dependent(true);
        break;
      }
      case LlvmObservationSpace::BUILDTIME: {
        space.mutable_double_sequence()->mutable_length_range()->mutable_min()->set_value(0);
        space.mutable_double_sequence()->mutable_scalar_range()->mutable_min()->set_value(0);
        space.set_deterministic(false);
        space.set_platform_dependent(true);
        break;
      }
      case LlvmObservationSpace::IS_BUILDABLE:
      case LlvmObservationSpace::IS_RUNNABLE: {
        auto featureSize = space.mutable_scalar_int64_range();
        featureSize->mutable_min()->set_value(0);
        featureSize->mutable_max()->set_value(1);
        space.set_deterministic(true);
        space.set_platform_dependent(true);
        space.mutable_default_value()->set_scalar_int64(0);
        break;
      }
    }
    spaces.push_back(space);
  }
  return spaces;
}

}  // namespace compiler_gym::llvm_service

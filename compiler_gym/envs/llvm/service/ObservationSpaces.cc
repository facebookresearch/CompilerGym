// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"

#include <glog/logging.h>
#include <stdint.h>

#include <limits>
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
  std::vector<ObservationSpace> observationSpaces;
  observationSpaces.reserve(magic_enum::enum_count<LlvmObservationSpace>());
  for (const auto& value : magic_enum::enum_values<LlvmObservationSpace>()) {
    ObservationSpace observationSpace;
    observationSpace.set_name(util::enumNameToPascalCase<LlvmObservationSpace>(value));
    Space& space = *observationSpace.mutable_space();
    switch (value) {
      case LlvmObservationSpace::IR: {
        space.mutable_string_value()->mutable_length_range()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::IR_SHA1: {
        space.mutable_string_value()->mutable_length_range()->set_min(40);
        space.mutable_string_value()->mutable_length_range()->set_max(40);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::BITCODE: {
        space.mutable_byte_sequence()->mutable_length_range()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::BITCODE_FILE: {
        space.mutable_string_value()->mutable_length_range()->set_min(0);
        // 4096 is the maximum path length for most filesystems.
        space.mutable_string_value()->mutable_length_range()->set_max(kMaximumPathLength);
        // A random file path is generated, so the returned value is not
        // deterministic.
        observationSpace.set_deterministic(false);
        observationSpace.set_platform_dependent(false);
        break;
      }
      case LlvmObservationSpace::INST_COUNT: {
        Int64Box& featureSizes = *space.mutable_int64_box();

        Int64Tensor& featureSizesLow = *featureSizes.mutable_low();
        *featureSizesLow.mutable_shape()->Add() = kInstCountFeatureDimensionality;
        std::vector<int64_t> low(kInstCountFeatureDimensionality, 0);
        featureSizesLow.mutable_value()->Add(low.begin(), low.end());

        Int64Tensor& featureSizesHigh = *featureSizes.mutable_high();
        *featureSizesHigh.mutable_shape()->Add() = kInstCountFeatureDimensionality;
        std::vector<int64_t> high(kInstCountFeatureDimensionality,
                                  std::numeric_limits<int64_t>::max());
        featureSizesHigh.mutable_value()->Add(high.begin(), high.end());

        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);

        *observationSpace.mutable_default_observation()
             ->mutable_int64_tensor()
             ->mutable_shape()
             ->Add() = kInstCountFeatureDimensionality;
        observationSpace.mutable_default_observation()
            ->mutable_int64_tensor()
            ->mutable_value()
            ->Add(low.begin(), low.end());
        break;
      }
      case LlvmObservationSpace::AUTOPHASE: {
        Int64Box& featureSizes = *space.mutable_int64_box();

        Int64Tensor& featureSizesLow = *featureSizes.mutable_low();
        *featureSizesLow.mutable_shape()->Add() = kAutophaseFeatureDim;
        std::vector<int64_t> low(kAutophaseFeatureDim, 0);
        featureSizesLow.mutable_value()->Add(low.begin(), low.end());

        Int64Tensor& featureSizesHigh = *featureSizes.mutable_high();
        *featureSizesHigh.mutable_shape()->Add() = kAutophaseFeatureDim;
        std::vector<int64_t> high(kAutophaseFeatureDim, std::numeric_limits<int64_t>::max());
        featureSizesHigh.mutable_value()->Add(high.begin(), high.end());

        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);

        *observationSpace.mutable_default_observation()
             ->mutable_int64_tensor()
             ->mutable_shape()
             ->Add() = kAutophaseFeatureDim;
        observationSpace.mutable_default_observation()
            ->mutable_int64_tensor()
            ->mutable_value()
            ->Add(low.begin(), low.end());
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
        space.mutable_string_value()->mutable_length_range()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);
        programl::ProgramGraph graph;
        json nodeLinkGraph;
        CHECK(programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph).ok())
            << "Failed to serialize default ProGraML graph";
        Opaque opaque;
        opaque.set_format("json://networkx/MultiDiGraph");
        *opaque.mutable_data() = nodeLinkGraph.dump();
        observationSpace.mutable_default_observation()->mutable_any_value()->PackFrom(opaque);
        break;
      }
      case LlvmObservationSpace::PROGRAML_JSON: {
        // ProGraML serializes the graph to JSON.
        space.mutable_string_value()->mutable_length_range()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);
        programl::ProgramGraph graph;
        json nodeLinkGraph;
        CHECK(programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph).ok())
            << "Failed to serialize default ProGraML graph";
        Opaque opaque;
        opaque.set_format("json://");
        *opaque.mutable_data() = nodeLinkGraph.dump();
        observationSpace.mutable_default_observation()->mutable_any_value()->PackFrom(opaque);
        break;
      }
      case LlvmObservationSpace::CPU_INFO: {
        // Hardware info is returned as a JSON
        space.mutable_string_value()->mutable_length_range()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(true);
        Opaque opaque;
        opaque.set_format("json://");
        *opaque.mutable_data() = "{}";
        observationSpace.mutable_default_observation()->mutable_any_value()->PackFrom(opaque);
        break;
      }
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3:
      case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ: {
        space.mutable_int64_value()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(false);
        observationSpace.mutable_default_observation()->set_int64_value(0);
        break;
      }
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_BYTES:
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_O0:
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_O3:
      case LlvmObservationSpace::OBJECT_TEXT_SIZE_OZ: {
        space.mutable_int64_value()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(true);
        observationSpace.mutable_default_observation()->set_int64_value(0);
        break;
      }
      case LlvmObservationSpace::TEXT_SIZE_BYTES:
      case LlvmObservationSpace::TEXT_SIZE_O0:
      case LlvmObservationSpace::TEXT_SIZE_O3:
      case LlvmObservationSpace::TEXT_SIZE_OZ: {
        space.mutable_int64_value()->set_min(0);
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(true);
        observationSpace.mutable_default_observation()->set_int64_value(0);
        break;
      }
      case LlvmObservationSpace::RUNTIME:
      case LlvmObservationSpace::BUILDTIME: {
        space.mutable_double_sequence()->mutable_length_range()->set_min(0);
        space.mutable_double_sequence()->mutable_scalar_range()->set_min(0);
        observationSpace.set_deterministic(false);
        observationSpace.set_platform_dependent(true);
        break;
      }
      case LlvmObservationSpace::IS_BUILDABLE:
      case LlvmObservationSpace::IS_RUNNABLE: {
        space.mutable_boolean_value();
        observationSpace.set_deterministic(true);
        observationSpace.set_platform_dependent(true);
        observationSpace.mutable_default_observation()->set_int64_value(0);
        break;
      }
    }
    observationSpaces.push_back(observationSpace);
  }
  return observationSpaces;
}

}  // namespace compiler_gym::llvm_service

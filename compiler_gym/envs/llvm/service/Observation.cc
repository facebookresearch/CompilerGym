// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/Observation.h"

#include <cpuinfo.h>
#include <glog/logging.h>

#include <iomanip>
#include <sstream>
#include <string>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/third_party/LexedIr/lexed_ir.h"
#include "compiler_gym/third_party/autophase/InstCount.h"
#include "compiler_gym/third_party/llvm/InstCount.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "llvm/Bitcode/BitcodeWriter.h"
// #include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "nlohmann/json.hpp"
#include "programl/graph/format/node_link_graph.h"
#include "programl/ir/llvm/llvm.h"

namespace fs = boost::filesystem;

namespace compiler_gym::llvm_service {

using grpc::Status;
using grpc::StatusCode;
using nlohmann::json;

const programl::ProgramGraphOptions programlOptions;

Status setObservation(LlvmObservationSpace space, const fs::path& workingDirectory,
                      Benchmark& benchmark, Event& reply) {
  switch (space) {
    case LlvmObservationSpace::IR: {
      // Serialize the LLVM module to an IR string.
      std::string ir;
      llvm::raw_string_ostream rso(ir);
      benchmark.module().print(rso, /*AAW=*/nullptr);
      rso.flush();
      reply.set_string_value(ir);
      break;
    }
    case LlvmObservationSpace::IR_SHA1: {
      std::stringstream ss;
      const BenchmarkHash hash = benchmark.module_hash();
      // Hex encode, zero pad, and concatenate the unsigned integers that
      // contain the hash.
      for (uint32_t val : hash) {
        ss << std::setfill('0') << std::setw(sizeof(BenchmarkHash::value_type) * 2) << std::hex
           << val;
      }
      reply.set_string_value(ss.str());
      break;
    }
    case LlvmObservationSpace::BITCODE: {
      std::string bitcode;
      llvm::raw_string_ostream outbuffer(bitcode);
      llvm::WriteBitcodeToFile(benchmark.module(), outbuffer);
      outbuffer.flush();
      *reply.mutable_byte_tensor()->mutable_shape()->Add() = bitcode.size();
      *reply.mutable_byte_tensor()->mutable_value() = bitcode;
      break;
    }
    case LlvmObservationSpace::BITCODE_FILE: {
      // Generate an output path with 16 bits of randomness.
      const auto outpath = fs::unique_path(workingDirectory / "module-%%%%%%%%.bc");
      RETURN_IF_ERROR(writeBitcodeFile(benchmark.module(), outpath.string()));
      reply.set_string_value(outpath.string());
      break;
    }
    case LlvmObservationSpace::INST_COUNT: {
      const auto features = InstCount::getFeatureVector(benchmark.module());
      *reply.mutable_int64_tensor()->mutable_shape()->Add() = features.size();
      *reply.mutable_int64_tensor()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::AUTOPHASE: {
      const auto features = autophase::InstCount::getFeatureVector(benchmark.module());
      *reply.mutable_int64_tensor()->mutable_shape()->Add() = features.size();
      *reply.mutable_int64_tensor()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::PROGRAML:
    case LlvmObservationSpace::PROGRAML_JSON: {
      // Build the ProGraML graph.
      programl::ProgramGraph graph;
      auto status =
          programl::ir::llvm::BuildProgramGraph(benchmark.module(), &graph, programlOptions);
      if (!status.ok()) {
        return Status(StatusCode::INTERNAL, status.error_message());
      }

      // Serialize the graph to a JSON node link graph.
      json nodeLinkGraph;
      status = programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph);
      if (!status.ok()) {
        return Status(StatusCode::INTERNAL, status.error_message());
      }
      Opaque opaque;
      opaque.set_format(space == LlvmObservationSpace::PROGRAML ? "json://networkx/MultiDiGraph"
                                                                : "json://");
      *opaque.mutable_data() = nodeLinkGraph.dump();
      reply.mutable_any_value()->PackFrom(opaque);
      break;
    }
    case LlvmObservationSpace::CPU_INFO: {
      json hwinfo;
      auto caches = {
          std::make_tuple("l1i_cache", cpuinfo_get_l1i_caches(), cpuinfo_get_l1d_caches_count()),
          {"l1d_cache", cpuinfo_get_l1d_caches(), cpuinfo_get_l1d_caches_count()},
          {"l2_cache", cpuinfo_get_l2_caches(), cpuinfo_get_l2_caches_count()},
          {"l3_cache", cpuinfo_get_l3_caches(), cpuinfo_get_l3_caches_count()},
          {"l4_cache", cpuinfo_get_l4_caches(), cpuinfo_get_l4_caches_count()}};
      for (auto [name, cache, count] : caches) {
        std::string sizeName = std::string(name) + "_size";
        std::string countName = std::string(name) + "_count";
        if (cache) {
          hwinfo[sizeName] = cache->size;
          hwinfo[countName] = count;
        } else {
          hwinfo[sizeName] = -1;
          hwinfo[countName] = count;
        }
      }
      hwinfo["cores_count"] = cpuinfo_get_cores_count();
      auto cpu = cpuinfo_get_packages();
      hwinfo["name"] = cpu->name;
      Opaque opaque;
      opaque.set_format("json://");
      *opaque.mutable_data() = hwinfo.dump();
      reply.mutable_any_value()->PackFrom(opaque);
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::IR_INSTRUCTION_COUNT, benchmark.module(),
                              workingDirectory, benchmark.dynamicConfig(), &cost));
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES, benchmark.module(),
                              workingDirectory, benchmark.dynamicConfig(), &cost));
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::TEXT_SIZE_BYTES, benchmark.module(),
                              workingDirectory, benchmark.dynamicConfig(), &cost));
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_int64_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::RUNTIME: {
      return benchmark.computeRuntime(reply);
    }
    case LlvmObservationSpace::IS_BUILDABLE: {
      reply.set_boolean_value(benchmark.isBuildable());
      break;
    }
    case LlvmObservationSpace::IS_RUNNABLE: {
      reply.set_boolean_value(benchmark.isRunnable());
      break;
    }
    case LlvmObservationSpace::BUILDTIME: {
      return benchmark.computeBuildtime(reply);
    }
    case LlvmObservationSpace::LEXED_IR: {
      // Serialize the LLVM module to an IR string.
      std::string ir;
      llvm::raw_string_ostream rso(ir);
      benchmark.module().print(rso, /*AAW=*/nullptr);
      rso.flush();

      const auto lexed = LexedIr::LexIR(ir);
      const auto token_id = lexed.first.first;
      const auto token_kind = lexed.first.second;
      const auto token_cat = lexed.second.first;
      const auto token_values = lexed.second.second;

      Event token_id_ev, token_kind_ev, token_cat_ev, token_values_ev;
      token_id_ev.mutable_int64_tensor()->add_shape(token_id.size());
      *token_id_ev.mutable_int64_tensor()->mutable_value() = {token_id.begin(), token_id.end()};

      token_kind_ev.mutable_string_tensor()->add_shape(token_kind.size());
      *token_kind_ev.mutable_string_tensor()->mutable_value() = {token_kind.begin(),
                                                                 token_kind.end()};

      token_cat_ev.mutable_string_tensor()->add_shape(token_cat.size());
      *token_cat_ev.mutable_string_tensor()->mutable_value() = {token_cat.begin(), token_cat.end()};

      token_values_ev.mutable_string_tensor()->add_shape(token_values.size());
      *token_values_ev.mutable_string_tensor()->mutable_value() = {token_values.begin(),
                                                                   token_values.end()};

      (*reply.mutable_event_dict()->mutable_event())["token_id"] = token_id_ev;
      (*reply.mutable_event_dict()->mutable_event())["token_kind"] = token_kind_ev;
      (*reply.mutable_event_dict()->mutable_event())["token_category"] = token_cat_ev;
      (*reply.mutable_event_dict()->mutable_event())["token_value"] = token_values_ev;
      break;
    }
  }

  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

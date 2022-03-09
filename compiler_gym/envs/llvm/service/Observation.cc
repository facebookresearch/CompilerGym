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
#include "compiler_gym/third_party/autophase/InstCount.h"
#include "compiler_gym/third_party/llvm/InstCount.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "llvm/Bitcode/BitcodeWriter.h"
// #include "llvm/IR/Metadata.h"
#include "IR2Vec.h"
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
                      Benchmark& benchmark, Observation& reply) {
  switch (space) {
    case LlvmObservationSpace::IR: {
      // Serialize the LLVM module to an IR string.
      std::string ir;
      llvm::raw_string_ostream rso(ir);
      benchmark.module().print(rso, /*AAW=*/nullptr);
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
      reply.set_binary_value(outbuffer.str());
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
      *reply.mutable_int64_list()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::AUTOPHASE: {
      const auto features = autophase::InstCount::getFeatureVector(benchmark.module());
      *reply.mutable_int64_list()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::IR2VEC_FLOW_AWARE: {
      const auto ir2vecEmbeddingsPath = util::getRunfilesPath(
          "compiler_gym/third_party/ir2vec/seedEmbeddingVocab-300-llvm10.txt");

      IR2Vec::Embeddings embeddings(benchmark.module(), IR2Vec::IR2VecMode::FlowAware,
                                    ir2vecEmbeddingsPath.string());
      const auto features = embeddings.getProgramVector();
      *reply.mutable_double_list()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::IR2VEC_SYMBOLIC: {
      const auto ir2vecEmbeddingsPath = util::getRunfilesPath(
          "compiler_gym/third_party/ir2vec/seedEmbeddingVocab-300-llvm10.txt");

      IR2Vec::Embeddings embeddings(benchmark.module(), IR2Vec::IR2VecMode::Symbolic,
                                    ir2vecEmbeddingsPath.string());
      const auto features = embeddings.getProgramVector();
      *reply.mutable_double_list()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::IR2VEC_FUNCTION_LEVEL_FLOW_AWARE: {
      const auto ir2vecEmbeddingsPath = util::getRunfilesPath(
          "compiler_gym/third_party/ir2vec/seedEmbeddingVocab-300-llvm10.txt");
      IR2Vec::Embeddings embeddings(benchmark.module(), IR2Vec::IR2VecMode::FlowAware,
                                    ir2vecEmbeddingsPath.string());
      const auto FuncMap = embeddings.getFunctionVecMap();
      json Embeddings = json::array({});

      for (auto func : FuncMap) {
        std::vector<double> FuncEmb = {func.second.begin(), func.second.end()};
        json FuncEmbJson = FuncEmb;
        json FuncJson;
        std::string FuncName = func.first->getName();
        FuncJson[FuncName] = FuncEmbJson;
        Embeddings.push_back(FuncJson);
      }
      *reply.mutable_string_value() = Embeddings.dump();
      break;
    }
    case LlvmObservationSpace::IR2VEC_FUNCTION_LEVEL_SYMBOLIC: {
      const auto ir2vecEmbeddingsPath = util::getRunfilesPath(
          "compiler_gym/third_party/ir2vec/seedEmbeddingVocab-300-llvm10.txt");
      IR2Vec::Embeddings embeddings(benchmark.module(), IR2Vec::IR2VecMode::Symbolic,
                                    ir2vecEmbeddingsPath.string());
      const auto FuncMap = embeddings.getFunctionVecMap();
      json Embeddings = json::array({});

      for (auto func : FuncMap) {
        std::vector<double> FuncEmb = {func.second.begin(), func.second.end()};
        json FuncEmbJson = FuncEmb;
        json FuncJson;
        std::string FuncName = func.first->getName();
        FuncJson[FuncName] = FuncEmbJson;
        Embeddings.push_back(FuncJson);
      }
      *reply.mutable_string_value() = Embeddings.dump();
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
      *reply.mutable_string_value() = nodeLinkGraph.dump();
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
      *reply.mutable_string_value() = hwinfo.dump();
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::IR_INSTRUCTION_COUNT, benchmark.module(),
                              workingDirectory, &cost));
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES, benchmark.module(),
                              workingDirectory, &cost));
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
    case LlvmObservationSpace::TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(
          setCost(LlvmCostFunction::TEXT_SIZE_BYTES, benchmark.module(), workingDirectory, &cost));
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark.baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
#endif
    case LlvmObservationSpace::RUNTIME: {
      return benchmark.computeRuntime(reply);
    }
    case LlvmObservationSpace::IS_BUILDABLE: {
      reply.set_scalar_int64(benchmark.isBuildable() ? 1 : 0);
      break;
    }
    case LlvmObservationSpace::IS_RUNNABLE: {
      reply.set_scalar_int64(benchmark.isRunnable() ? 1 : 0);
      break;
    }
    case LlvmObservationSpace::BUILDTIME: {
      return benchmark.computeBuildtime(reply);
    }
  }

  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

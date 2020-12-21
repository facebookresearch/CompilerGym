// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmEnvironment.h"

#include <cpuinfo.h>
#include <fmt/format.h>
#include <glog/logging.h>

#include <optional>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/envs/llvm/service/passes/ActionHeaders.h"
#include "compiler_gym/envs/llvm/service/passes/ActionSwitch.h"
#include "compiler_gym/third_party/autophase/InstCount.h"
#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "nlohmann/json.hpp"
#include "programl/graph/format/node_link_graph.h"
#include "programl/ir/llvm/llvm.h"

namespace fs = boost::filesystem;

namespace compiler_gym::llvm_service {

using grpc::Status;
using grpc::StatusCode;
using nlohmann::json;

namespace {

// Return the target library information for a module.
llvm::TargetLibraryInfoImpl getTargetLibraryInfo(llvm::Module& module) {
  llvm::Triple triple(module.getTargetTriple());
  return llvm::TargetLibraryInfoImpl(triple);
}

// Wrapper around llvm::verifyModule() which raises the given exception type
// on failure.
Status verifyModuleStatus(const llvm::Module& module) {
  std::string errorMessage;
  llvm::raw_string_ostream rso(errorMessage);
  if (llvm::verifyModule(module, &rso)) {
    rso.flush();
    return Status(StatusCode::INTERNAL, "Failed to verify module: " + errorMessage);
  }
  return Status::OK;
}

void initLlvm() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Initialize passes.
  llvm::PassRegistry& Registry = *llvm::PassRegistry::getPassRegistry();
  llvm::initializeCore(Registry);
  llvm::initializeCoroutines(Registry);
  llvm::initializeScalarOpts(Registry);
  llvm::initializeObjCARCOpts(Registry);
  llvm::initializeVectorization(Registry);
  llvm::initializeIPO(Registry);
  llvm::initializeAnalysis(Registry);
  llvm::initializeTransformUtils(Registry);
  llvm::initializeInstCombine(Registry);
  llvm::initializeAggressiveInstCombine(Registry);
  llvm::initializeInstrumentation(Registry);
  llvm::initializeTarget(Registry);
  llvm::initializeExpandMemCmpPassPass(Registry);
  llvm::initializeScalarizeMaskedMemIntrinPass(Registry);
  llvm::initializeCodeGenPreparePass(Registry);
  llvm::initializeAtomicExpandPass(Registry);
  llvm::initializeRewriteSymbolsLegacyPassPass(Registry);
  llvm::initializeWinEHPreparePass(Registry);
  llvm::initializeDwarfEHPreparePass(Registry);
  llvm::initializeSafeStackLegacyPassPass(Registry);
  llvm::initializeSjLjEHPreparePass(Registry);
  llvm::initializePreISelIntrinsicLoweringLegacyPassPass(Registry);
  llvm::initializeGlobalMergePass(Registry);
  llvm::initializeIndirectBrExpandPassPass(Registry);
  llvm::initializeInterleavedAccessPass(Registry);
  llvm::initializeEntryExitInstrumenterPass(Registry);
  llvm::initializePostInlineEntryExitInstrumenterPass(Registry);
  llvm::initializeUnreachableBlockElimLegacyPassPass(Registry);
  llvm::initializeExpandReductionsPass(Registry);
  llvm::initializeWasmEHPreparePass(Registry);
  llvm::initializeWriteBitcodePassPass(Registry);
}

}  // anonymous namespace

LlvmEnvironment::LlvmEnvironment(std::unique_ptr<Benchmark> benchmark, LlvmActionSpace actionSpace,
                                 std::optional<LlvmObservationSpace> eagerObservationSpace,
                                 std::optional<LlvmRewardSpace> eagerRewardSpace,
                                 const boost::filesystem::path& workingDirectory)
    : workingDirectory_(workingDirectory),
      benchmark_(std::move(benchmark)),
      actionSpace_(actionSpace),
      eagerObservationSpace_(eagerObservationSpace),
      eagerRewardSpace_(eagerRewardSpace),
      tlii_(getTargetLibraryInfo(benchmark_->module())),
      actionCount_(0) {
  // Initialize LLVM.
  initLlvm();

  // Initialize cpuinfo
  cpuinfo_initialize();

  // Strip module debug info.
  llvm::StripDebugInfo(benchmark_->module());

  // Erase module-level named metadata.
  while (!benchmark_->module().named_metadata_empty()) {
    llvm::NamedMDNode* nmd = &*benchmark_->module().named_metadata_begin();
    benchmark_->module().eraseNamedMetadata(nmd);
  }

  // Verify the module now to catch any problems early.
  CHECK(verifyModuleStatus(benchmark_->module()).ok());

  // Compute initial eager observation and reward if required.
  // TODO(cummins): Defer these so that we can replace CHECKs with status codes.
  if (eagerObservationSpace_.has_value()) {
    CHECK(getObservation(eagerObservationSpace_.value(), &eagerObservation_).ok());
  }
  if (eagerRewardSpace_.has_value()) {
    CHECK(getReward(eagerRewardSpace_.value(), &eagerReward_).ok());
  }
}

Status LlvmEnvironment::takeAction(const ActionRequest& request, ActionReply* reply) {
  actionCount_ += request.action_size();
  switch (actionSpace()) {
    case LlvmActionSpace::PASSES_ALL:
      for (int i = 0; i < request.action_size(); ++i) {
        LlvmAction action;
        RETURN_IF_ERROR(util::intToEnum(request.action(i), &action));
// Use the generated HANDLE_PASS() switch statement to dispatch to runPass().
#define HANDLE_PASS(pass) runPass(pass, reply);
        HANDLE_ACTION(action, HANDLE_PASS)
#undef HANDLE_PASS
      }
  }

  // Fail now if we have broken something.
  RETURN_IF_ERROR(verifyModuleStatus(benchmark().module()));

  if (eagerObservationSpace().has_value()) {
    // Compute new observation if needed.
    if (!reply->action_had_no_effect()) {
      eagerObservation_ = {};
      RETURN_IF_ERROR(getObservation(eagerObservationSpace().value(), &eagerObservation_));
    }
    *reply->mutable_observation() = eagerObservation_;
  }

  if (eagerRewardSpace().has_value()) {
    if (reply->action_had_no_effect()) {
      // Action had no effect, so no reward.
      reply->mutable_reward()->set_reward(0);
    } else {
      // Compute new reward if needed.
      eagerReward_ = {};
      RETURN_IF_ERROR(getReward(eagerRewardSpace().value(), &eagerReward_));
      *reply->mutable_reward() = eagerReward_;
    }
  }

  return Status::OK;
}

void LlvmEnvironment::runPass(llvm::Pass* pass, ActionReply* reply) {
  llvm::legacy::PassManager passManager;
  setupPassManager(&passManager, pass);

  const bool changed = passManager.run(benchmark().module());
  reply->set_action_had_no_effect(!changed);
}

void LlvmEnvironment::runPass(llvm::FunctionPass* pass, ActionReply* reply) {
  llvm::legacy::FunctionPassManager passManager(&benchmark().module());
  setupPassManager(&passManager, pass);

  bool changed = passManager.doInitialization();
  for (auto& function : benchmark().module()) {
    changed |= (passManager.run(function) ? 1 : 0);
  }
  changed |= (passManager.doFinalization() ? 1 : 0);
  reply->set_action_had_no_effect(!changed);
}

Status LlvmEnvironment::getObservation(LlvmObservationSpace space, Observation* reply) {
  switch (space) {
    case LlvmObservationSpace::IR: {
      // Serialize the LLVM module to an IR string.
      std::string ir;
      llvm::raw_string_ostream rso(ir);
      benchmark().module().print(rso, /*AAW=*/nullptr);
      reply->set_string_value(ir);
      break;
    }
    case LlvmObservationSpace::BITCODE_FILE: {
      // Generate an output path with 16 bits of randomness.
      const std::string outpath =
          fs::unique_path(workingDirectory_ / "module-%%%%%%%%.bc").string();
      std::error_code error;
      llvm::raw_fd_ostream outfile(outpath, error);
      if (error.value()) {
        return Status(StatusCode::INTERNAL,
                      fmt::format("Failed to write bitcode file: {}", outpath));
      }
      llvm::WriteBitcodeToFile(benchmark().module(), outfile);
      reply->set_string_value(outpath);
      break;
    }
    case LlvmObservationSpace::AUTOPHASE: {
      const auto features = autophase::InstCount::getFeatureVector(benchmark().module());
      *reply->mutable_int64_list()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::PROGRAML: {
      // Build the ProGraML graph.
      programl::ProgramGraph graph;
      auto status =
          programl::ir::llvm::BuildProgramGraph(benchmark().module(), &graph, programlOptions_);
      if (!status.ok()) {
        return Status(StatusCode::INTERNAL, status.error_message());
      }

      // Serialize the graph to a JSON node link graph.
      json nodeLinkGraph;
      status = programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph);
      if (!status.ok()) {
        return Status(StatusCode::INTERNAL, status.error_message());
      }
      *reply->mutable_string_value() = nodeLinkGraph.dump();
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
      *reply->mutable_string_value() = hwinfo.dump();
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT: {
      const auto cost =
          getCost(LlvmCostFunction::IR_INSTRUCTION_COUNT, benchmark().module(), workingDirectory_);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES: {
      const auto cost = getCost(LlvmCostFunction::NATIVE_TEXT_SIZE_BYTES, benchmark().module(),
                                workingDirectory_);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::NATIVE_TEXT_SIZE_BYTES);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::NATIVE_TEXT_SIZE_BYTES);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::NATIVE_TEXT_SIZE_BYTES_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::NATIVE_TEXT_SIZE_BYTES);
      reply->mutable_int64_list()->add_value(static_cast<int64_t>(cost));
      break;
    }
  }

  return Status::OK;
}

Status LlvmEnvironment::getReward(LlvmRewardSpace space, Reward* reply) {
  const LlvmCostFunction cost = getCostFunction(space);
  const auto costIdx = static_cast<size_t>(cost);
  const LlvmBaselinePolicy baselinePolicy = getBaselinePolicy(space);

  // Fetch the cached costs.
  const double unoptimizedCost =
      getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0, cost);
  const double baselineCost = getBaselineCost(benchmark().baselineCosts(), baselinePolicy, cost);
  const double previousCost =
      previousCosts_[costIdx].has_value() ? *previousCosts_[costIdx] : unoptimizedCost;

  // Compute a new cost.
  const double currentCost = getCost(cost, benchmark().module(), workingDirectory_);

  // Derive the reward from the costs.
  double reward = previousCost - currentCost;
  if (baselinePolicy != LlvmBaselinePolicy::O0) {
    reward /= unoptimizedCost - baselineCost;
  }
  reply->set_reward(reward);

  // Update the cached costs.
  previousCosts_[costIdx] = currentCost;

  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

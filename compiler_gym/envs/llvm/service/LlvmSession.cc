// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmSession.h"

#include <cpuinfo.h>
#include <fmt/format.h>
#include <glog/logging.h>

#include <optional>
#include <subprocess/subprocess.hpp>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/envs/llvm/service/passes/ActionHeaders.h"
#include "compiler_gym/envs/llvm/service/passes/ActionSwitch.h"
#include "compiler_gym/third_party/autophase/InstCount.h"
#include "compiler_gym/third_party/llvm/InstCount.h"
#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
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
    return Status(StatusCode::DATA_LOSS, "Failed to verify module: " + errorMessage);
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

Status writeBitcodeToFile(const llvm::Module& module, const fs::path& path) {
  std::error_code error;
  llvm::raw_fd_ostream outfile(path.string(), error);
  if (error.value()) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to write bitcode file: {}", path.string()));
  }
  llvm::WriteBitcodeToFile(module, outfile);
  return Status::OK;
}

}  // anonymous namespace

LlvmSession::LlvmSession(std::unique_ptr<Benchmark> benchmark, LlvmActionSpace actionSpace,
                         const boost::filesystem::path& workingDirectory)
    : workingDirectory_(workingDirectory),
      benchmark_(std::move(benchmark)),
      actionSpace_(actionSpace),
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
}

Status LlvmSession::step(const StepRequest& request, StepReply* reply) {
  // Apply the requested actions.
  actionCount_ += request.action_size();
  switch (actionSpace()) {
    case LlvmActionSpace::PASSES_ALL:
      for (int i = 0; i < request.action_size(); ++i) {
        LlvmAction action;
        RETURN_IF_ERROR(util::intToEnum(request.action(i), &action));
        RETURN_IF_ERROR(runAction(action, reply));
      }
  }

  // Fail now if we have broken something.
  RETURN_IF_ERROR(verifyModuleStatus(benchmark().module()));

  // Compute the requested observations.
  for (int i = 0; i < request.observation_space_size(); ++i) {
    LlvmObservationSpace observationSpace;
    RETURN_IF_ERROR(util::intToEnum(request.observation_space(i), &observationSpace));
    auto observation = reply->add_observation();
    RETURN_IF_ERROR(getObservation(observationSpace, observation));
  }

  return Status::OK;
}

Status LlvmSession::runAction(LlvmAction action, StepReply* reply) {
#ifdef EXPERIMENTAL_UNSTABLE_GVN_SINK_PASS
  // NOTE(https://github.com/facebookresearch/CompilerGym/issues/46): The
  // -gvn-sink pass has been found to have nondeterministic behavior so has
  // been disabled in compiler_gym/envs/llvm/service/pass/config.py. Invoking
  // the command line was found to produce more stable results.
  if (action == LlvmAction::GVNSINK_PASS) {
    RETURN_IF_ERROR(runOptWithArgs({"-gvn-sink"}));
    reply->set_action_had_no_effect(true);
    return Status::OK;
  }
#endif

// Use the generated HANDLE_PASS() switch statement to dispatch to runPass().
#define HANDLE_PASS(pass) runPass(pass, reply);
  HANDLE_ACTION(action, HANDLE_PASS)
#undef HANDLE_PASS

  return Status::OK;
}

void LlvmSession::runPass(llvm::Pass* pass, StepReply* reply) {
  llvm::legacy::PassManager passManager;
  setupPassManager(&passManager, pass);

  const bool changed = passManager.run(benchmark().module());
  reply->set_action_had_no_effect(!changed);
}

void LlvmSession::runPass(llvm::FunctionPass* pass, StepReply* reply) {
  llvm::legacy::FunctionPassManager passManager(&benchmark().module());
  setupPassManager(&passManager, pass);

  bool changed = passManager.doInitialization();
  for (auto& function : benchmark().module()) {
    changed |= (passManager.run(function) ? 1 : 0);
  }
  changed |= (passManager.doFinalization() ? 1 : 0);
  reply->set_action_had_no_effect(!changed);
}

Status LlvmSession::runOptWithArgs(const std::vector<std::string>& optArgs) {
  // Create temporary files for `opt` to read from and write to.
  const auto before_path = fs::unique_path(workingDirectory_ / "module-%%%%%%%%.bc");
  const auto after_path = fs::unique_path(workingDirectory_ / "module-%%%%%%%%.bc");
  RETURN_IF_ERROR(writeBitcodeToFile(benchmark().module(), before_path));

  // Build a command line invocation: `opt input.bc -o output.bc <optArgs...>`.
  const auto optPath = util::getSiteDataPath("llvm-v0/bin/opt");
  if (!fs::exists(optPath)) {
    return Status(StatusCode::INTERNAL, fmt::format("File not found: {}", optPath.string()));
  }
  std::vector<std::string> optCmd{optPath.string(), before_path.string(), "-o",
                                  after_path.string()};
  optCmd.insert(optCmd.end(), optArgs.begin(), optArgs.end());

  // Run the opt command line.
  auto opt = subprocess::Popen(optCmd, subprocess::output{subprocess::PIPE},
                               subprocess::error{subprocess::PIPE});
  const auto optOutput = opt.communicate();
  fs::remove(before_path);

  if (opt.retcode()) {
    if (fs::exists(after_path)) {
      fs::remove(after_path);
    }
    const std::string error(optOutput.second.buf.begin(), optOutput.second.buf.end());
    return Status(StatusCode::INTERNAL, error);
  }

  if (!fs::exists(after_path)) {
    return Status(StatusCode::INTERNAL, "Failed to generate output file");
  }

  // Read the bitcode file generated by `opt`.
  Bitcode bitcode;
  auto status = readBitcodeFile(after_path, &bitcode);
  fs::remove(after_path);
  if (!status.ok()) {
    return status;
  }

  // Replace the benchmark's module with the one generated by `opt`.
  auto module = makeModule(benchmark().context(), bitcode, benchmark().name(), &status);
  RETURN_IF_ERROR(status);
  benchmark().replaceModule(std::move(module));

  return Status::OK;
}

Status LlvmSession::getObservation(LlvmObservationSpace space, Observation* reply) {
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
      const auto outpath = fs::unique_path(workingDirectory_ / "module-%%%%%%%%.bc");
      RETURN_IF_ERROR(writeBitcodeToFile(benchmark().module(), outpath));
      reply->set_string_value(outpath.string());
      break;
    }
    case LlvmObservationSpace::INST_COUNT: {
      const auto features = InstCount::getFeatureVector(benchmark().module());
      *reply->mutable_int64_list()->mutable_value() = {features.begin(), features.end()};
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
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::IR_INSTRUCTION_COUNT, benchmark().module(),
                              workingDirectory_, &cost));
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES, benchmark().module(),
                              workingDirectory_, &cost));
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
    case LlvmObservationSpace::TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::TEXT_SIZE_BYTES, benchmark().module(),
                              workingDirectory_, &cost));
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply->set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
#endif
  }

  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

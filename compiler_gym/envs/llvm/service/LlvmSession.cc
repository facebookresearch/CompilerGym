// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmSession.h"

#include <cpuinfo.h>
#include <fmt/format.h>
#include <glog/logging.h>
#include <stdio.h>
#include <stdlib.h>

#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <subprocess/subprocess.hpp>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/envs/llvm/service/passes/ActionHeaders.h"
#include "compiler_gym/envs/llvm/service/passes/ActionSwitch.h"
#include "compiler_gym/third_party/autophase/InstCount.h"
#include "compiler_gym/third_party/llvm/InstCount.h"
#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/Subprocess.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
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

using BenchmarkProto = compiler_gym::Benchmark;
using ActionSpaceProto = compiler_gym::ActionSpace;

namespace {

std::string exec(const char* cmd) {
  char buffer[128];
  std::string result = "";
  FILE* pipe = popen(cmd, "r");
  if (!pipe)
    throw std::runtime_error("popen() failed!");
  try {
    while (fgets(buffer, sizeof buffer, pipe) != NULL) {
      result += buffer;
    }
  } catch (...) {
    pclose(pipe);
    throw;
  }
  pclose(pipe);
  return result;
}

// Return the target library information for a module.
llvm::TargetLibraryInfoImpl getTargetLibraryInfo(llvm::Module& module) {
  llvm::Triple triple(module.getTargetTriple());
  return llvm::TargetLibraryInfoImpl(triple);
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

std::string LlvmSession::getCompilerVersion() const {
  std::stringstream ss;
  ss << LLVM_VERSION_STRING << " " << llvm::Triple::normalize(LLVM_DEFAULT_TARGET_TRIPLE);
  return ss.str();
}

std::vector<ActionSpace> LlvmSession::getActionSpaces() const { return getLlvmActionSpaceList(); }

std::vector<ObservationSpace> LlvmSession::getObservationSpaces() const {
  return getLlvmObservationSpaceList();
}

LlvmSession::LlvmSession(const boost::filesystem::path& workingDirectory)
    : CompilationSession(workingDirectory),
      observationSpaceNames_(util::createPascalCaseToEnumLookupTable<LlvmObservationSpace>()) {
  cpuinfo_initialize();
}

Status LlvmSession::init(const ActionSpace& actionSpace, const BenchmarkProto& benchmark) {
  BenchmarkFactory& benchmarkFactory = BenchmarkFactory::getSingleton(workingDirectory());

  // Get the benchmark or return an error.
  std::unique_ptr<Benchmark> llvmBenchmark;
  RETURN_IF_ERROR(benchmarkFactory.getBenchmark(benchmark, &llvmBenchmark));

  // Verify the benchmark now to catch errors early.
  RETURN_IF_ERROR(llvmBenchmark->verify_module());

  LlvmActionSpace actionSpaceEnum;
  RETURN_IF_ERROR(util::pascalCaseToEnum(actionSpace.name(), &actionSpaceEnum));

  return init(actionSpaceEnum, std::move(llvmBenchmark));
}

Status LlvmSession::init(CompilationSession* other) {
  // TODO: Static cast?
  auto llvmOther = static_cast<LlvmSession*>(other);
  return init(llvmOther->actionSpace(), llvmOther->benchmark().clone(workingDirectory()));
}

Status LlvmSession::init(const LlvmActionSpace& actionSpace, std::unique_ptr<Benchmark> benchmark) {
  benchmark_ = std::move(benchmark);
  actionSpace_ = actionSpace;

  tlii_ = getTargetLibraryInfo(benchmark_->module());

  return Status::OK;
}

Status LlvmSession::applyAction(const Action& action, bool& endOfEpisode,
                                std::optional<ActionSpace>& newActionSpace,
                                bool& actionHadNoEffect) {
  DCHECK(benchmark_) << "Calling applyAction() before init()";

  // Apply the requested action.
  switch (actionSpace()) {
    case LlvmActionSpace::PASSES_ALL:
      LlvmAction actionEnum;
      RETURN_IF_ERROR(util::intToEnum(action.action(), &actionEnum));
      RETURN_IF_ERROR(applyPassAction(actionEnum, actionHadNoEffect));
  }

  return Status::OK;
}

Status LlvmSession::endOfStep(bool actionHadNoEffect, bool& endOfEpisode,
                              std::optional<ActionSpace>& newActionSpace) {
  if (actionHadNoEffect) {
    return Status::OK;
  } else {
    return benchmark().verify_module();
  }
}

Status LlvmSession::computeObservation(const ObservationSpace& observationSpace,
                                       Observation& observation) {
  DCHECK(benchmark_) << "Calling computeObservation() before init()";

  const auto& it = observationSpaceNames_.find(observationSpace.name());
  if (it == observationSpaceNames_.end()) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        fmt::format("Could not interpret observation space name: {}", observationSpace.name()));
  }
  const LlvmObservationSpace observationSpaceEnum = it->second;
  RETURN_IF_ERROR(computeObservation(observationSpaceEnum, observation));
  return Status::OK;
}

// We could use this function to pass the bambu synthesized function
Status LlvmSession::handleSessionParameter(const std::string& key, const std::string& value,
                                           std::optional<std::string>& reply) {
  if (key == "llvm.set_runtimes_per_observation_count") {
    const int ivalue = std::stoi(value);
    if (ivalue < 1) {
      return Status(
          StatusCode::INVALID_ARGUMENT,
          fmt::format("runtimes_per_observation_count must be >= 1. Received: {}", ivalue));
    }
    benchmark().setRuntimesPerObservationCount(ivalue);
    reply = value;
  } else if (key == "llvm.get_runtimes_per_observation_count") {
    reply = fmt::format("{}", benchmark().getRuntimesPerObservationCount());
  } else if (key == "llvm.set_buildtimes_per_observation_count") {
    const int ivalue = std::stoi(value);
    if (ivalue < 1) {
      return Status(
          StatusCode::INVALID_ARGUMENT,
          fmt::format("buildtimes_per_observation_count must be >= 1. Received: {}", ivalue));
    }
    benchmark().setBuildtimesPerObservationCount(ivalue);
    reply = value;
  } else if (key == "llvm.get_buildtimes_per_observation_count") {
    reply = fmt::format("{}", benchmark().getBuildtimesPerObservationCount());
  } else if (key == "llvm.apply_baseline_optimizations") {
    if (value == "-Oz") {
      bool changed = benchmark().applyBaselineOptimizations(/*optLevel=*/2, /*sizeLevel=*/2);
      reply = changed ? "1" : "0";
    } else if (value == "-O3") {
      bool changed = benchmark().applyBaselineOptimizations(/*optLevel=*/3, /*sizeLevel=*/0);
      reply = changed ? "1" : "0";
    } else {
      return Status(StatusCode::INVALID_ARGUMENT,
                    fmt::format("Invalid value for llvm.apply_baseline_optimizations: {}", value));
    }
  }
  return Status::OK;
}

Status LlvmSession::applyPassAction(LlvmAction action, bool& actionHadNoEffect) {
#ifdef EXPERIMENTAL_UNSTABLE_GVN_SINK_PASS
  // NOTE(https://github.com/facebookresearch/CompilerGym/issues/46): The
  // -gvn-sink pass has been found to have nondeterministic behavior so has
  // been disabled in compiler_gym/envs/llvm/service/pass/config.py. Invoking
  // the command line was found to produce more stable results.
  if (action == LlvmAction::GVNSINK_PASS) {
    RETURN_IF_ERROR(runOptWithArgs({"-gvn-sink"}));
    actionHadNoEffect = true;
    return Status::OK;
  }
#endif

// Use the generated HANDLE_PASS() switch statement to dispatch to runPass().
#define HANDLE_PASS(pass) actionHadNoEffect = !runPass(pass);
  HANDLE_ACTION(action, HANDLE_PASS)
#undef HANDLE_PASS

  if (!actionHadNoEffect) {
    benchmark().markModuleModified();
  }

  return Status::OK;
}

bool LlvmSession::runPass(llvm::Pass* pass) {
  llvm::legacy::PassManager passManager;
  setupPassManager(&passManager, pass);

  return passManager.run(benchmark().module());
}

bool LlvmSession::runPass(llvm::FunctionPass* pass) {
  llvm::legacy::FunctionPassManager passManager(&benchmark().module());
  setupPassManager(&passManager, pass);

  bool changed = passManager.doInitialization();
  for (auto& function : benchmark().module()) {
    changed |= (passManager.run(function) ? 1 : 0);
  }
  changed |= (passManager.doFinalization() ? 1 : 0);
  return changed;
}

Status LlvmSession::runOptWithArgs(const std::vector<std::string>& optArgs) {
  // Create temporary files for `opt` to read from and write to.
  const auto before_path = fs::unique_path(workingDirectory() / "module-%%%%%%%%.bc");
  const auto after_path = fs::unique_path(workingDirectory() / "module-%%%%%%%%.bc");
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

Status LlvmSession::computeObservation(LlvmObservationSpace space, Observation& reply) {
  switch (space) {
    case LlvmObservationSpace::IR: {
      // Serialize the LLVM module to an IR string.
      std::string ir;
      llvm::raw_string_ostream rso(ir);
      benchmark().module().print(rso, /*AAW=*/nullptr);
      reply.set_string_value(ir);
      break;
    }
    case LlvmObservationSpace::IR_SHA1: {
      std::stringstream ss;
      const BenchmarkHash hash = benchmark().module_hash();
      // Hex encode, zero pad, and concatenate the unsigned integers that
      // contain the hash.
      for (uint32_t val : hash) {
        ss << std::setfill('0') << std::setw(sizeof(BenchmarkHash::value_type) * 2) << std::hex
           << val;
      }
      reply.set_string_value(ss.str());
      break;
    }
    case LlvmObservationSpace::BITCODE_FILE: {
      // Generate an output path with 16 bits of randomness.
      const auto outpath = fs::unique_path(workingDirectory() / "module-%%%%%%%%.bc");
      RETURN_IF_ERROR(writeBitcodeToFile(benchmark().module(), outpath));
      reply.set_string_value(outpath.string());
      break;
    }
    case LlvmObservationSpace::INST_COUNT: {
      const auto features = InstCount::getFeatureVector(benchmark().module());
      *reply.mutable_int64_list()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::AUTOPHASE: {
      const auto features = autophase::InstCount::getFeatureVector(benchmark().module());
      *reply.mutable_int64_list()->mutable_value() = {features.begin(), features.end()};
      break;
    }
    case LlvmObservationSpace::PROGRAML:
    case LlvmObservationSpace::PROGRAML_JSON: {
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
      RETURN_IF_ERROR(setCost(LlvmCostFunction::IR_INSTRUCTION_COUNT, benchmark().module(),
                              workingDirectory(), &cost));
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::IR_INSTRUCTION_COUNT_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::IR_INSTRUCTION_COUNT);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES, benchmark().module(),
                              workingDirectory(), &cost));
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::OBJECT_TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::OBJECT_TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
    case LlvmObservationSpace::TEXT_SIZE_BYTES: {
      double cost;
      RETURN_IF_ERROR(setCost(LlvmCostFunction::TEXT_SIZE_BYTES, benchmark().module(),
                              workingDirectory(), &cost));
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O0: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O0,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_O3: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::O3,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
    case LlvmObservationSpace::TEXT_SIZE_OZ: {
      const auto cost = getBaselineCost(benchmark().baselineCosts(), LlvmBaselinePolicy::Oz,
                                        LlvmCostFunction::TEXT_SIZE_BYTES);
      reply.set_scalar_int64(static_cast<int64_t>(cost));
      break;
    }
#endif
    case LlvmObservationSpace::RUNTIME: {
      return benchmark().computeRuntime(reply);
    }
    case LlvmObservationSpace::IS_BUILDABLE: {
      reply.set_scalar_int64(benchmark().isBuildable() ? 1 : 0);
      break;
    }
    case LlvmObservationSpace::IS_RUNNABLE: {
      reply.set_scalar_int64(benchmark().isRunnable() ? 1 : 0);
      break;
    }
    case LlvmObservationSpace::BUILDTIME: {
      return benchmark().computeBuildtime(reply);
    }
    case LlvmObservationSpace::CIRCUIT_AREA: {
      constexpr int kBambuTimeoutSeconds = 60;
      std::string stdout, stdout2, stdoutdis, stdoutecho;
      util::checkOutput(">&2 echo hola > /tmp/hola_output.txt 2>&1", kBambuTimeoutSeconds,
                        workingDirectory(), stdoutecho);
      // Write the bitcode to a file.
      RETURN_IF_ERROR(writeBitcodeToFile(benchmark().module(), workingDirectory() / "bambu.bc"));

      // Run bambu on the bitcode and record the stdout.
      // util::checkOutput(fmt::format("du {}", (workingDirectory() / "bambu.bc").string()),
      //                  kBambuTimeoutSeconds, workingDirectory(), stdout);

      // for tmp_synthesis.txt we might need to add the working directory
      util::checkOutput(
          fmt::format("llvm-dis-10 {} -o {}", (workingDirectory() / "bambu.bc").string(),
                      (workingDirectory() / "bambu.ll").string()),
          kBambuTimeoutSeconds, workingDirectory(), stdoutdis);

      // std::string exec(const char* cmd) {
      // std::string cmd_out =
      // exec(fmt::format("/home/ibrumar/tools/panda-github-install-llvm10-qm/bin/bambu {}
      // --compiler=I386_CLANG10 --top-fname=compare -v 4 > /tmp/bambu_out.txt 2>&1" ,
      // (workingDirectory() / "bambu.ll").string()).c_str()); std::cout << "\nThe bambu1 command
      // output is \n" << cmd_out << "\n";

      auto errCode = util::checkOutput(
          fmt::format(
              "/home/ibrumar/tools/panda-github-install-llvm10-qm/bin/bambu {} "
              "--compiler=I386_CLANG10 --top-fname=compare -v 4 > /tmp/tmp_synthesis.txt 2>&1",
              (workingDirectory() / "bambu.ll").string()),
          kBambuTimeoutSeconds, workingDirectory(), stdout);
      // std::ofstream myfile;
      // myfile.open ((workingDirectory() / "/tmp/tmp_synthesis.txt").string());
      // myfile << stdout;
      // myfile.close();

      auto errCode2 =
          util::checkOutput(fmt::format("grep \"Total estimated area\" /tmp/tmp_synthesis.txt | "
                                        "tail -n 1 | grep \"[0-9]*\" -o"),
                            kBambuTimeoutSeconds, workingDirectory(), stdout2);
      //      VLOG(1) << "\nOutput of the bambu command \n" << stdout << "\n and output of grep " <<
      //      stdout2 << "\n";
      // std::cout << "\nOutput of the bambu command \n" << stdout << "\n and output of grep " <<
      // stdout2 << "\n";

      // std::cout << "The error code is " << errCode.error_message() << "\n";
      // Parse the output of bambu.
      // TODO(ibrumar): Actually parse the output of bambu, including error
      // handling.
      int area = atoi(stdout2.c_str());
      if ((not errCode.ok()) or (not errCode2.ok()))
        area = 9999999;
      std::cout << "Area converted to integer is " << area << "\n";
      std::cout << "Bambu execution status was  " << errCode.ok() << "and grep exec status "
                << errCode2.ok() << "\n";
      reply.set_scalar_int64(area);
      break;
    }
  }

  return Status::OK;
}

}  // namespace compiler_gym::llvm_service

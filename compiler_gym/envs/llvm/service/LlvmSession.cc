// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/LlvmSession.h"

#include <cpuinfo.h>
#include <fmt/format.h>
#include <glog/logging.h>

#include <boost/process.hpp>
#include <chrono>
#include <future>
#include <iomanip>
#include <optional>
#include <string>

#include "boost/filesystem.hpp"
#include "compiler_gym/envs/llvm/service/ActionSpace.h"
#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/envs/llvm/service/Cost.h"
#include "compiler_gym/envs/llvm/service/Observation.h"
#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"
#include "compiler_gym/envs/llvm/service/passes/10.0.0/ActionHeaders.h"
#include "compiler_gym/envs/llvm/service/passes/10.0.0/ActionSwitch.h"
#include "compiler_gym/third_party/autophase/InstCount.h"
#include "compiler_gym/third_party/llvm/InstCount.h"
#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
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
namespace bp = boost::process;

namespace compiler_gym::llvm_service {

using grpc::Status;
using grpc::StatusCode;
using nlohmann::json;

using BenchmarkProto = compiler_gym::Benchmark;
using ActionSpaceProto = compiler_gym::ActionSpace;

namespace {

// Return the target library information for a module.
llvm::TargetLibraryInfoImpl getTargetLibraryInfo(llvm::Module& module) {
  llvm::Triple triple(module.getTargetTriple());
  return llvm::TargetLibraryInfoImpl(triple);
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

Status LlvmSession::applyAction(const Event& action, bool& endOfEpisode,
                                std::optional<ActionSpace>& newActionSpace,
                                bool& actionHadNoEffect) {
  DCHECK(benchmark_) << "Calling applyAction() before init()";

  // Apply the requested action.
  switch (actionSpace()) {
    case LlvmActionSpace::PASSES_ALL:
      LlvmAction actionEnum;
      if (action.value_case() != Event::ValueCase::kInt64Value) {
        return Status(StatusCode::INVALID_ARGUMENT,
                      fmt::format("Invalid action. Expected {}, received {}.",
                                  magic_enum::enum_name(Event::ValueCase::kInt64Value),
                                  magic_enum::enum_name(action.value_case())));
      }
      RETURN_IF_ERROR(util::intToEnum(action.int64_value(), &actionEnum));
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
                                       Event& observation) {
  DCHECK(benchmark_) << "Calling computeObservation() before init()";

  const auto& it = observationSpaceNames_.find(observationSpace.name());
  if (it == observationSpaceNames_.end()) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        fmt::format("Could not interpret observation space name: {}", observationSpace.name()));
  }
  const LlvmObservationSpace observationSpaceEnum = it->second;

  return setObservation(observationSpaceEnum, workingDirectory(), benchmark(), observation);
}

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
  } else if (key == "llvm.set_warmup_runs_count_per_runtime_observation") {
    const int ivalue = std::stoi(value);
    if (ivalue < 0) {
      return Status(
          StatusCode::INVALID_ARGUMENT,
          fmt::format("warmup_runs_count_per_runtime_observation must be >= 0. Received: {}",
                      ivalue));
    }
    benchmark().setWarmupRunsPerRuntimeObservationCount(ivalue);
    reply = value;
  } else if (key == "llvm.get_warmup_runs_count_per_runtime_observation") {
    reply = fmt::format("{}", benchmark().getWarmupRunsPerRuntimeObservationCount());
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
  RETURN_IF_ERROR(writeBitcodeFile(benchmark().module(), before_path));

  // Build a command line invocation: `opt input.bc -o output.bc <optArgs...>`.
  const auto optPath = util::getSiteDataPath("llvm-v0/bin/opt");
  if (!fs::exists(optPath)) {
    return Status(StatusCode::INTERNAL, fmt::format("File not found: {}", optPath.string()));
  }

  std::string optCmd =
      fmt::format("{} {} -o {}", optPath.string(), before_path.string(), after_path.string());
  for (const auto& arg : optArgs) {
    optCmd += " " + arg;
  }

  // Run the opt command line.
  try {
    boost::asio::io_context optStderrStream;
    std::future<std::string> optStderrFuture;

    bp::child opt(optCmd, bp::std_in.close(), bp::std_out > bp::null, bp::std_err > optStderrFuture,
                  optStderrStream);

    if (!util::wait_for(opt, std::chrono::seconds(60))) {
      return Status(StatusCode::DEADLINE_EXCEEDED,
                    fmt::format("Failed to run opt within 60 seconds: {}", optCmd));
    }
    optStderrStream.run();
    if (opt.exit_code()) {
      const std::string stderr = optStderrFuture.get();
      return Status(StatusCode::INTERNAL,
                    fmt::format("Opt command '{}' failed with return code {}: {}", optCmd,
                                opt.exit_code(), stderr));
    }
    fs::remove(before_path);
  } catch (bp::process_error& e) {
    fs::remove(before_path);
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to run opt command '{}': {}", optCmd, e.what()));
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

}  // namespace compiler_gym::llvm_service

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/mlir/service/MlirSession.h"

#include <cpuinfo.h>
#include <fmt/format.h>
#include <glog/logging.h>

#include <boost/process.hpp>
#include <chrono>
#include <future>
#include <iomanip>
#include <optional>
#include <string>

#include "boost/algorithm/string.hpp"
#include "boost/filesystem.hpp"
#include "compiler_gym/envs/mlir/service/ActionSpace.h"
#include "compiler_gym/envs/mlir/service/Benchmark.h"
#include "compiler_gym/envs/mlir/service/BenchmarkFactory.h"
#include "compiler_gym/envs/mlir/service/MlirUtils.h"
#include "compiler_gym/envs/mlir/service/Observation.h"
#include "compiler_gym/envs/mlir/service/ObservationSpaces.h"
#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "nlohmann/json.hpp"

namespace fs = boost::filesystem;
namespace bp = boost::process;

namespace compiler_gym::mlir_service {

using grpc::Status;
using grpc::StatusCode;
using nlohmann::json;

using BenchmarkProto = compiler_gym::Benchmark;
using ActionSpaceProto = compiler_gym::ActionSpace;

std::string MlirSession::getCompilerVersion() const {
  std::stringstream ss;
  // MLIR doesn't maintain a version as it lives in the llvm-project.
  ss << "LLVM " << LLVM_VERSION_STRING << " patch=" << LLVM_VERSION_PATCH << " "
     << llvm::Triple::normalize(LLVM_DEFAULT_TARGET_TRIPLE);
  return ss.str();
}

std::vector<ActionSpace> MlirSession::getActionSpaces() const { return getMlirActionSpaceList(); }

std::vector<ObservationSpace> MlirSession::getObservationSpaces() const {
  return getMlirObservationSpaceList();
}

MlirSession::MlirSession(const boost::filesystem::path& workingDirectory)
    : CompilationSession(workingDirectory),
      observationSpaceNames_(util::createPascalCaseToEnumLookupTable<MlirObservationSpace>()),
      spaceContainsEventChecker_(makeDefaultSpaceContainsEventChecker()) {
  cpuinfo_initialize();
}

Status MlirSession::init(const ActionSpace& actionSpace, const BenchmarkProto& benchmark) {
  BenchmarkFactory& benchmarkFactory = BenchmarkFactory::getSingleton(workingDirectory());

  // Get the benchmark or return an error.
  std::unique_ptr<Benchmark> mlirBenchmark;
  RETURN_IF_ERROR(benchmarkFactory.getBenchmark(benchmark, &mlirBenchmark));

  // Verify the benchmark now to catch errors early.
  RETURN_IF_ERROR(mlirBenchmark->verify_module());

  MlirActionSpace actionSpaceEnum;
  RETURN_IF_ERROR(util::pascalCaseToEnum(actionSpace.name(), &actionSpaceEnum));

  return init(actionSpaceEnum, std::move(mlirBenchmark));
}

Status MlirSession::init(CompilationSession* other) {
  // TODO: Static cast?
  auto mlirOther = static_cast<MlirSession*>(other);
  return init(mlirOther->actionSpace(), mlirOther->benchmark().clone(workingDirectory()));
}

Status MlirSession::init(const MlirActionSpace& actionSpace, std::unique_ptr<Benchmark> benchmark) {
  benchmark_ = std::move(benchmark);
  actionSpace_ = actionSpace;
  actionSpaces_ = this->getActionSpaces();
  actions_count_ = 0;

  return Status::OK;
}

Status MlirSession::applyAction(const Event& action, bool& endOfEpisode,
                                std::optional<ActionSpace>& newActionSpace,
                                bool& actionHadNoEffect) {
  ++actions_count_;
  endOfEpisode = true;
  DCHECK(benchmark_) << "Calling applyAction() before init()";

  spaceContainsEventChecker_.checkContains(actionSpaces_[0].space(), action);
  return mlir::performLinalgCodegen(action, benchmark().module());
}

Status MlirSession::endOfStep(bool actionHadNoEffect, bool& endOfEpisode,
                              std::optional<ActionSpace>& newActionSpace) {
  if (actionHadNoEffect) {
    return Status::OK;
  } else {
    return benchmark().verify_module();
  }
}

Status MlirSession::computeObservation(const ObservationSpace& observationSpace,
                                       Event& observation) {
  DCHECK(benchmark_) << "Calling computeObservation() before init()";

  const auto& it = observationSpaceNames_.find(observationSpace.name());
  if (it == observationSpaceNames_.end()) {
    return Status(
        StatusCode::INVALID_ARGUMENT,
        fmt::format("Could not interpret observation space name: {}", observationSpace.name()));
  }
  const MlirObservationSpace observationSpaceEnum = it->second;

  if (observationSpaceEnum == MlirObservationSpace::RUNTIME && actions_count_ == 0) {
    // Do nothing before first action.
    *observation.mutable_double_tensor()->mutable_value()->Add() = 0;
    *observation.mutable_double_tensor()->mutable_shape()->Add() = 1;
    return Status::OK;
  } else {
    return setObservation(observationSpaceEnum, workingDirectory(), benchmark(), observation);
  }
}

Status MlirSession::handleSessionParameter(const std::string& key, const std::string& value,
                                           std::optional<std::string>& reply) {
  if (key == "mlir.set_runtimes_per_observation_count") {
    const int ivalue = std::stoi(value);
    if (ivalue < 1) {
      return Status(
          StatusCode::INVALID_ARGUMENT,
          fmt::format("runtimes_per_observation_count must be >= 1. Received: {}", ivalue));
    }
    benchmark().setRuntimesPerObservationCount(ivalue);
    reply = value;
  } else if (key == "mlir.get_runtimes_per_observation_count") {
    reply = fmt::format("{}", benchmark().getRuntimesPerObservationCount());
  } else if (key == "mlir.set_warmup_runs_count_per_runtime_observation") {
    const int ivalue = std::stoi(value);
    if (ivalue < 0) {
      return Status(
          StatusCode::INVALID_ARGUMENT,
          fmt::format("warmup_runs_count_per_runtime_observation must be >= 0. Received: {}",
                      ivalue));
    }
    benchmark().setWarmupRunsPerRuntimeObservationCount(ivalue);
    reply = value;
  } else if (key == "mlir.get_warmup_runs_count_per_runtime_observation") {
    reply = fmt::format("{}", benchmark().getWarmupRunsPerRuntimeObservationCount());
  } else if (key == "mlir.set_buildtimes_per_observation_count") {
    const int ivalue = std::stoi(value);
    if (ivalue < 1) {
      return Status(
          StatusCode::INVALID_ARGUMENT,
          fmt::format("buildtimes_per_observation_count must be >= 1. Received: {}", ivalue));
    }
    benchmark().setBuildtimesPerObservationCount(ivalue);
    reply = value;
  } else if (key == "mlir.get_buildtimes_per_observation_count") {
    reply = fmt::format("{}", benchmark().getBuildtimesPerObservationCount());
  }
  return Status::OK;
}

}  // namespace compiler_gym::mlir_service

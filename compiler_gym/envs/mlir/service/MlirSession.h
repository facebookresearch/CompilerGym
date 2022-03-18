// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <magic_enum.hpp>
#include <memory>
#include <optional>
#include <unordered_map>

#include "compiler_gym/envs/mlir/service/ActionSpace.h"
#include "compiler_gym/envs/mlir/service/Benchmark.h"
#include "compiler_gym/envs/mlir/service/Observation.h"
#include "compiler_gym/envs/mlir/service/ObservationSpaces.h"
#include "compiler_gym/service/CompilationSession.h"
#include "compiler_gym/service/proto/Proto.h"
#include "compiler_gym/service/proto/compiler_gym_service.grpc.pb.h"
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

namespace compiler_gym::mlir_service {

/**
 * An interactive MLIR compilation session.
 *
 * This class exposes the MLIR optimization pipeline for an MLIR module as an
 * interactive environment. It can be used directly as a C++ API, or it can be
 * accessed through an RPC interface using the CompilerGym RPC runtime.
 */
class MlirSession final : public CompilationSession {
 public:
  MlirSession(const boost::filesystem::path& workingDirectory);

  std::string getCompilerVersion() const final override;

  std::vector<ActionSpace> getActionSpaces() const final override;

  std::vector<ObservationSpace> getObservationSpaces() const final override;

  [[nodiscard]] grpc::Status init(const ActionSpace& actionSpace,
                                  const compiler_gym::Benchmark& benchmark) final override;

  [[nodiscard]] grpc::Status init(CompilationSession* other) final override;

  [[nodiscard]] grpc::Status applyAction(const Event& action, bool& endOfEpisode,
                                         std::optional<ActionSpace>& newActionSpace,
                                         bool& actionHadNoEffect) final override;

  [[nodiscard]] grpc::Status endOfStep(bool actionHadNoEffect, bool& endOfEpisode,
                                       std::optional<ActionSpace>& newActionSpace) final override;

  [[nodiscard]] grpc::Status computeObservation(const ObservationSpace& observationSpace,
                                                Event& observation) final override;

  [[nodiscard]] virtual grpc::Status handleSessionParameter(
      const std::string& key, const std::string& value,
      std::optional<std::string>& reply) final override;

  inline const MlirActionSpace actionSpace() const { return actionSpace_; }

 private:
  [[nodiscard]] grpc::Status computeObservation(MlirObservationSpace observationSpace,
                                                Event& observation);

  [[nodiscard]] grpc::Status init(const MlirActionSpace& actionSpace,
                                  std::unique_ptr<Benchmark> benchmark);

  inline const Benchmark& benchmark() const {
    DCHECK(benchmark_) << "Calling benchmark() before init()";
    return *benchmark_;
  }
  inline Benchmark& benchmark() {
    DCHECK(benchmark_) << "Calling benchmark() before init()";
    return *benchmark_;
  }

  // Immutable state.
  const std::unordered_map<std::string, MlirObservationSpace> observationSpaceNames_;
  // Mutable state initialized in init().
  MlirActionSpace actionSpace_;
  std::vector<ActionSpace> actionSpaces_;
  std::unique_ptr<Benchmark> benchmark_;
  unsigned actions_count_;
  SpaceContainsEventChecker spaceContainsEventChecker_;
};

}  // namespace compiler_gym::mlir_service

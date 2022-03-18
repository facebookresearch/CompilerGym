// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/mlir/service/MlirSession.h"
#include "compiler_gym/service/runtime/Runtime.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/ArmNeon/ArmNeonDialect.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Quant/Passes.h"
#include "mlir/Dialect/SCF/Passes.h"
#include "mlir/Dialect/SPIRV/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tosa/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Transforms/Passes.h"
//#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/GPU/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Quant/QuantOps.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/IR/Dialect.h"

const char* usage = R"(MLIR CompilerGym service)";

using namespace compiler_gym::runtime;
using namespace compiler_gym::mlir_service;

namespace mlir {

namespace {
void initMlir(int argc, char** argv) {
  // TODO(kyleherndon)

  // Core Transforms
  registerCanonicalizerPass();
  registerCSEPass();
  registerInlinerPass();
  registerLocationSnapshotPass();
  registerLoopCoalescingPass();
  registerLoopInvariantCodeMotionPass();
  registerAffineScalarReplacementPass();
  registerSCFParallelLoopCollapsingPass();
  registerPrintOpStatsPass();
  registerViewOpGraphPass();
  registerStripDebugInfoPass();
  registerSymbolDCEPass();

  // Generic conversions
  // registerReconcileUnrealizedCastsPass();

  // Affine
  registerAffinePasses();
  registerAffineLoopFusionPass();
  registerAffinePipelineDataTransferPass();
  registerConvertAffineToStandardPass();

  // Linalg
  registerLinalgPasses();

  // LLVM
  registerConvertArmNeon2dToIntrPass();

  // MemRef
  // memref::registerMemRefPasses();

  // SCF
  registerSCFParallelLoopFusionPass();
  registerSCFParallelLoopTilingPass();
  registerSCFToStandardPass();

  // Quant
  // quant::registerQuantPasses();

  // Shape
  // registerShapePasses();

  // SPIR-V
  spirv::registerSPIRVLowerABIAttributesPass();
  registerConvertGPUToSPIRVPass();
  registerConvertStandardToSPIRVPass();
  registerConvertLinalgToSPIRVPass();

  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();
}

}  // anonymous namespace

}  // namespace mlir

int main(int argc, char** argv) {
  mlir::initMlir(argc, argv);
  return createAndRunCompilerGymService<MlirSession>(argc, argv, usage);
}

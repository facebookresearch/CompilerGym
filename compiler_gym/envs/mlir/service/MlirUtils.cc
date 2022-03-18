#include "compiler_gym/envs/mlir/service/MlirUtils.h"

#include <fmt/format.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>

#include <stdexcept>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
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

using grpc::Status;
using grpc::StatusCode;

using compiler_gym::Event;

namespace mlir {

// Register Dialects
void registerDialects(DialectRegistry* registry) {
  mlir::registerAllDialects(*registry);
  mlir::registerLLVMDialectTranslation(*registry);
}

// Init Registry
DialectRegistry createDialectRegistry() {
  mlir::DialectRegistry registry;
  registerDialects(&registry);
  return registry;
}

// Make Context
std::unique_ptr<MLIRContext> createMlirContext() {
  DialectRegistry registry = createDialectRegistry();

  std::unique_ptr<MLIRContext> context = std::make_unique<mlir::MLIRContext>(registry);
  context->loadAllAvailableDialects();
  return context;
}

// Lower to LLVM
Status lowerMLIRModuleToLLVM(OwningModuleRef& mlirModule, MLIRContext* context,
                             llvm::raw_string_ostream& moduleString) {
  ModuleOp module = *mlirModule;
  PassManager pm(context, mlir::OpPassManager::Nesting::Implicit);
  mlir::applyPassManagerCLOptions(pm);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLinalgComprehensiveModuleBufferizePass());

  // Lower to LLVM
  pm.addPass(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertLinalgToLoopsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createConvertLinalgToLLVMPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(mlir::createMemRefToLLVMPass());
  pm.addPass(mlir::createLowerToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());

  if (failed(pm.run(module))) {
    return Status(StatusCode::INTERNAL, "Error compiling mlir to llvm backend");
  }

  mlir::registerLLVMDialectTranslation(*module->getContext());
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
  if (!llvmModule) {
    return Status(StatusCode::INTERNAL, "Error translating to llvm ir");
  }
  moduleString << *llvmModule;
  return Status::OK;
}

// Linalg Codegen
namespace {
struct LinalgCodegenPass
    : public mlir::PassWrapper<LinalgCodegenPass, mlir::OperationPass<mlir::FuncOp>> {
  LinalgCodegenPass(const Event& e) : action(e) {}
  LinalgCodegenPass(const LinalgCodegenPass& pass) : action(pass.action) {}

  void getDependentDialects(mlir::DialectRegistry& registry) const override {
    // clang-format off
    registry.insert<mlir::AffineDialect,
                    mlir::linalg::LinalgDialect,
                    mlir::memref::MemRefDialect,
                    mlir::scf::SCFDialect,
                    mlir::StandardOpsDialect,
                    mlir::vector::VectorDialect>();
    // clang-format on
  }

  template <typename LinalgNamedOp>
  void applyStrategyToNamedLinalgOp();

  void runOnOperation() override;
  const Event& action;

  void runStrategy(const Event& action, StringRef anchorOpName);
};

void performTileOptions(const Event& action, mlir::linalg::LinalgTilingOptions& tilingOptions,
                        bool& promote, bool& promote_full_tile) {
  const Event& tileOptionsAction = action.event_dict().event().at("tile_options");
  const Event& tileSizesAction = tileOptionsAction.event_dict().event().at("tile_sizes");
  const Event& interchangeVectorAction =
      tileOptionsAction.event_dict().event().at("interchange_vector");
  const Event& promoteAction = tileOptionsAction.event_dict().event().at("promote");
  const Event& promoteFullTileAction =
      tileOptionsAction.event_dict().event().at("promote_full_tile");
  const Event& loopTypeAction = tileOptionsAction.event_dict().event().at("loop_type");

  mlir::linalg::LinalgTilingLoopType loop_type;
  if (loopTypeAction.int64_value() == 0) {
    loop_type = mlir::linalg::LinalgTilingLoopType::Loops;
  } else if (loopTypeAction.int64_value() == 1) {
    loop_type = mlir::linalg::LinalgTilingLoopType::AffineLoops;
  } else {
    CHECK(false) << fmt::format("Unexpected loop_type \"{}\".", loopTypeAction.int64_value());
  }

  CHECK(tileSizesAction.has_int64_tensor());
  llvm::SmallVector<int64_t, 4> tileSizes(tileSizesAction.int64_tensor().value().begin(),
                                          tileSizesAction.int64_tensor().value().end());

  CHECK(interchangeVectorAction.has_int64_tensor());
  llvm::SmallVector<int64_t, 4> tileInterchange(
      interchangeVectorAction.int64_tensor().value().begin(),
      interchangeVectorAction.int64_tensor().value().end());

  tilingOptions = tilingOptions.setLoopType(loop_type);
  tilingOptions = tilingOptions.setTileSizes(tileSizes);
  tilingOptions = tilingOptions.setInterchange(
      SmallVector<unsigned>(tileInterchange.begin(), tileInterchange.end()));

  promote = promoteAction.boolean_value();
  promote_full_tile = promoteFullTileAction.boolean_value();
}

void performVectorizeOptions(const Event& action,
                             mlir::vector::VectorContractLowering& vectorContractLowering,
                             mlir::vector::VectorTransferSplit& vectorTransferSplit,
                             bool& unrollVectorTransfers) {
  const Event& vectorizeOptionsAction = action.event_dict().event().at("vectorize_options");
  const Event& vectorizeToAction = vectorizeOptionsAction.event_dict().event().at("vectorize_to");
  const Event& vectorizeTransferSplitAction =
      vectorizeOptionsAction.event_dict().event().at("vector_transfer_split");
  const Event& unrollVectorTransfersAction =
      vectorizeOptionsAction.event_dict().event().at("unroll_vector_transfers");
  // Vectorize Codegen Options
  if (vectorizeToAction.int64_value() == 0) {
    vectorContractLowering = mlir::vector::VectorContractLowering::Dot;
  } else if (vectorizeToAction.int64_value() == 1) {
    vectorContractLowering = mlir::vector::VectorContractLowering::Matmul;
  } else if (vectorizeToAction.int64_value() == 2) {
    vectorContractLowering = mlir::vector::VectorContractLowering::OuterProduct;
  } else {
    CHECK(false) << fmt::format("Unexpected vectorize_to \"{}\".", vectorizeToAction.int64_value());
  }

  if (vectorizeTransferSplitAction.int64_value() == 0) {
    vectorTransferSplit = mlir::vector::VectorTransferSplit::None;
  } else if (vectorizeTransferSplitAction.int64_value() == 1) {
    vectorTransferSplit = mlir::vector::VectorTransferSplit::LinalgCopy;
  } else if (vectorizeTransferSplitAction.int64_value() == 2) {
    vectorTransferSplit = mlir::vector::VectorTransferSplit::VectorTransfer;
  } else {
    CHECK(false) << fmt::format("Unexpected vector_transfer_split \"{}\".",
                                vectorizeTransferSplitAction.int64_value());
  }

  unrollVectorTransfers = unrollVectorTransfersAction.boolean_value();
}

void LinalgCodegenPass::runStrategy(const Event& action, StringRef anchorOpName) {
  mlir::linalg::CodegenStrategy strategy;
  mlir::linalg::LinalgTilingOptions tilingOptions;
  mlir::vector::VectorContractLowering vectorContractLowering;
  mlir::vector::VectorTransferSplit vectorTransferSplit;
  bool unrollVectorTransfers, promote, promote_full_tile;

  performTileOptions(action, tilingOptions, promote, promote_full_tile);
  performVectorizeOptions(action, vectorContractLowering, vectorTransferSplit,
                          unrollVectorTransfers);

  strategy.tile(anchorOpName, tilingOptions)
      .promoteIf(
          promote, anchorOpName,
          mlir::linalg::LinalgPromotionOptions().setAlignment(16).setUseFullTileBuffersByDefault(
              promote_full_tile))
      .vectorize(anchorOpName)
      .vectorLowering(
          mlir::linalg::LinalgVectorLoweringOptions()
              .setVectorTransformsOptions(mlir::vector::VectorTransformsOptions()
                                              .setVectorTransformsOptions(vectorContractLowering)
                                              .setVectorTransferSplit(vectorTransferSplit))
              .setVectorTransferToSCFOptions(
                  mlir::VectorTransferToSCFOptions().enableFullUnroll(unrollVectorTransfers))
              .enableTransferPartialRewrite()
              .enableContractionLowering()
              .enableTransferToSCFConversion());
  // Created a nested OpPassManager and run.
  mlir::FuncOp funcOp = getOperation();
  mlir::OpPassManager dynamicPM("builtin.func");
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  if (failed(runPipeline(dynamicPM, funcOp)))
    return signalPassFailure();
}

void LinalgCodegenPass::runOnOperation() {
  // TODO(kyleherndon): figure out why this won't compile/remove it
  // mlir::MLIRContext *ctx = getOperation().getContext();
  // llvm::SmallVector<Attribute, 4> attrs;
  // attrs.push_back(mlir::ArrayAttr::get(ctx,
  //                                {mlir::StringAttr::get(ctx, "prefer-vector-width"),
  //                                 mlir::StringAttr::get(ctx, "256")}
  //                               ));
  // attrs.push_back(mlir::ArrayAttr::get(ctx,
  //                                {mlir::StringAttr::get(ctx, "target-cpu"),
  //                                 mlir::StringAttr::get(ctx, "haswell")}
  //                               ));
  // getOperation()->setAttr("passthrough", mlir::ArrayAttr::get(ctx, attrs));

  for (const Event& e : action.event_list().event()) {
    runStrategy(e, "linalg.matmul");
  }
}

std::unique_ptr<mlir::OperationPass<mlir::FuncOp>> createLinalgCodegenPass(const Event& action) {
  return std::make_unique<LinalgCodegenPass>(action);
}

}  // namespace

// Do action
Status performLinalgCodegen(const Event& action, OwningModuleRef& module) {
  mlir::ModuleOp moduleOp = *module;
  mlir::PassManager pm(moduleOp.getContext(), mlir::OpPassManager::Nesting::Implicit);

  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(createLinalgCodegenPass(action));
  pm.addPass(mlir::createLinalgComprehensiveModuleBufferizePass());

  if (failed(pm.run(moduleOp))) {
    return Status(StatusCode::INTERNAL, "Failure running pass");
  }
  return Status::OK;
}

}  // namespace mlir

#pragma once

#include <grpcpp/grpcpp.h>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

class ModuleOp;

// TODO(kyleherndon): docstrings
void registerDialects(mlir::DialectRegistry* registry);

DialectRegistry createDialectRegistry();

std::unique_ptr<MLIRContext> createMlirContext();

grpc::Status lowerMLIRModuleToLLVM(OwningOpRef<mlir::ModuleOp>& mlirModule, MLIRContext* context,
                                   llvm::raw_string_ostream& moduleString);

grpc::Status performLinalgCodegen(const compiler_gym::Event& action,
                                  OwningOpRef<mlir::ModuleOp>& module);

}  // namespace mlir

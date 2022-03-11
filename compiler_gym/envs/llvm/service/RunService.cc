// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/envs/llvm/service/LlvmSession.h"
#include "compiler_gym/service/runtime/Runtime.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/TargetSelect.h"

const char* usage = R"(LLVM CompilerGym service)";

using namespace compiler_gym::runtime;
using namespace compiler_gym::llvm_service;

namespace {

void initLlvm() {
  llvm::InitializeNativeTarget();

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

int main(int argc, char** argv) {
  initLlvm();
  const auto ret = createAndRunCompilerGymService<LlvmSession>(argc, argv, usage);

  // NOTE(github.com/facebookresearch/CompilerGym/issues/582): We need to make
  // sure that BenchmarkFactory::close() is called on the global singleton
  // instance, so that the temporary scratch directories are tidied up.
  //
  // TODO(github.com/facebookresearch/CompilerGym/issues/591): Once the runtime
  // has been refactored to support intra-session mutable state, this singleton
  // can be replaced by a member variable that is closed on
  // CompilerGymServiceContext::shutdown().
  BenchmarkFactory::getSingleton(FLAGS_working_dir).close();

  return ret;
}

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <algorithm>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopAccessAnalysis.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/Debugify.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Vectorize.h"

using namespace llvm;

namespace {
/// Input LLVM module file name.
cl::opt<std::string> InputFilename(cl::Positional, cl::desc("Specify input filename"),
                                   cl::value_desc("filename"), cl::init("-"));
/// Output LLVM module file name.
cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                    cl::value_desc("filename"), cl::init("-"));

/// Loop Optimizations
static cl::opt<bool> UnrollEnable("floop-unroll", cl::desc("Enable loop unrolling"),
                                  cl::init(false));

static cl::opt<unsigned> UnrollCount(
    "funroll-count", cl::desc("Use this unroll count for all loops including those with "
                              "unroll_count pragma values, for testing purposes"));

static cl::opt<bool> VectorizeEnable("floop-vectorize", cl::desc("Enable loop vectorize"),
                                     cl::init("false"));

static cl::opt<unsigned> VectorizationFactor("fforce-vector-width",
                                             cl::desc("Sets the SIMD width. Zero is autoselect."));

// Force binary on terminals
static cl::opt<bool> Force("f", cl::desc("Enable binary output on terminals"));

// Output assembly
static cl::opt<bool> OutputAssembly("S", cl::desc("Write output as LLVM assembly"));

// Preserve use list order
static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-uselistorder", cl::desc("Preserve use-list order when writing LLVM bitcode."),
    cl::init(true), cl::Hidden);

static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-uselistorder", cl::desc("Preserve use-list order when writing LLVM assembly."),
    cl::init(false), cl::Hidden);

// added from opt.cpp
static cl::opt<bool> DebugifyEach(
    "debugify-each", cl::desc("Start each pass with debugify and end it with check-debugify"));

class OptCustomPassManager : public legacy::PassManager {
  DebugifyStatsMap DIStatsMap;

 public:
  using super = legacy::PassManager;

  void add(Pass* P) override {
    // Wrap each pass with (-check)-debugify passes if requested, making
    // exceptions for passes which shouldn't see -debugify instrumentation.
    bool WrapWithDebugify =
        DebugifyEach && !P->getAsImmutablePass() && !isIRPrintingPass(P) && !isBitcodeWriterPass(P);
    if (!WrapWithDebugify) {
      super::add(P);
      return;
    }

    // Apply -debugify/-check-debugify before/after each pass and collect
    // debug info loss statistics.
    PassKind Kind = P->getPassKind();
    StringRef Name = P->getPassName();

    // TODO: Implement Debugify for LoopPass.
    switch (Kind) {
      case PT_Function:
        super::add(createDebugifyFunctionPass());
        super::add(P);
        super::add(createCheckDebugifyFunctionPass(true, Name, &DIStatsMap));
        break;
      case PT_Module:
        super::add(createDebugifyModulePass());
        super::add(P);
        super::add(createCheckDebugifyModulePass(true, Name, &DIStatsMap));
        break;
      default:
        super::add(P);
        break;
    }
  }

  const DebugifyStatsMap& getDebugifyStatsMap() const { return DIStatsMap; }
};

class LoopCounter : public llvm::FunctionPass {
 public:
  static char ID;
  std::unordered_map<std::string, int> counts;

  LoopCounter() : FunctionPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(llvm::Function& F) override {
    LoopInfo& LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto Loops = LI.getLoopsInPreorder();

    // Should really account for module, too.
    counts[F.getName().str()] = Loops.size();

    return false;
  }
};

char LoopCounter::ID = 0;

class LoopConfiguratorPass : public llvm::FunctionPass {
 public:
  static char ID;

  LoopConfiguratorPass() : FunctionPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(llvm::Function& F) override {
    LoopInfo& LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto Loops = LI.getLoopsInPreorder();

    // Should really account for module, too.
    for (auto ALoop : Loops) {
      if (UnrollEnable)
        addStringMetadataToLoop(ALoop, "llvm.loop.unroll.enable", UnrollEnable);
      if (UnrollCount)
        addStringMetadataToLoop(ALoop, "llvm.loop.unroll.count", UnrollCount);
      if (VectorizeEnable)
        addStringMetadataToLoop(ALoop, "llvm.loop.vectorize.enable", VectorizeEnable);
      if (VectorizationFactor)
        addStringMetadataToLoop(ALoop, "llvm.loop.vectorize.width", VectorizationFactor);
    }

    return false;
  }
};

char LoopConfiguratorPass::ID = 1;

/// Reads a module from a file.
/// On error, messages are written to stderr and null is returned.
///
/// \param Context LLVM Context for the module.
/// \param Name Input file name.
static std::unique_ptr<Module> readModule(LLVMContext& Context, StringRef Name) {
  SMDiagnostic Diag;
  std::unique_ptr<Module> Module = parseIRFile(Name, Diag, Context);

  if (!Module)
    Diag.print("llvm-counter", errs());

  return Module;
}

}  // namespace

namespace llvm {
// The INITIALIZE_PASS_XXX macros put the initialiser in the llvm namespace.
void initializeLoopCounterPass(PassRegistry& Registry);
void initializeLoopConfiguratorPassPass(PassRegistry& Registry);
}  // namespace llvm

// Initialise the pass. We have to declare the dependencies we use.
INITIALIZE_PASS_BEGIN(LoopCounter, "count-loops", "Count loops", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopCounter, "count-loops", "Count loops", false, false)

INITIALIZE_PASS_BEGIN(LoopConfiguratorPass, "unroll-loops-configurator",
                      "Configurates loop unrolling", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopConfiguratorPass, "unroll-loops-configurator",
                    "Configurates loop unrolling", false, false)

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              " LLVM-Counter\n\n"
                              " Count the loops in a bitcode file.\n");

  LLVMContext Context;
  SMDiagnostic Err;
  SourceMgr SM;
  std::error_code EC;

  std::unique_ptr<Module> Module = readModule(Context, InputFilename);

  if (!Module)
    return 1;

  // Prepare output
  ToolOutputFile Out(OutputFilename, EC, sys::fs::OF_None);
  if (EC) {
    Err = SMDiagnostic(OutputFilename, SourceMgr::DK_Error,
                       "Could not open output file: " + EC.message());
    Err.print(argv[0], errs());
    return 1;
  }

  initializeLoopCounterPass(*PassRegistry::getPassRegistry());
  OptCustomPassManager PM;
  LoopCounter* Counter = new LoopCounter();
  LoopConfiguratorPass* LoopConfigurator = new LoopConfiguratorPass();
  PM.add(Counter);
  PM.add(LoopConfigurator);
  PM.add(createLoopUnrollPass());
  PM.add(createLICMPass());
  PM.add(createLoopVectorizePass(false, false));
  PassManagerBuilder Builder;
  Builder.LoopVectorize = VectorizeEnable;
  Builder.populateModulePassManager(PM);

  // PM to output the module
  if (OutputAssembly) {
    PM.add(createPrintModulePass(Out.os(), "", PreserveAssemblyUseListOrder));
  } else if (Force || !CheckBitcodeOutputToConsole(Out.os())) {
    PM.add(createBitcodeWriterPass(Out.os(), PreserveBitcodeUseListOrder));
  }
  PM.run(*Module);

  // Log loop stats
  for (auto& x : Counter->counts) {
    llvm::dbgs() << x.first << ": " << x.second << " loops" << '\n';
  }

  Out.keep();

  return 0;
}

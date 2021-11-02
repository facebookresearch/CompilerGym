#include <algorithm>
#include <vector>

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
#define DEBUG_TYPE "LoopUnroller"

class LoopUnroller : public llvm::ModulePass {
 public:
  static char ID;

  LoopUnroller() : ModulePass(ID) {}

  bool runOnModule(llvm::Module& M) override {
    loopcounter = 0;
    for (auto IT = M.begin(), END = M.end(); IT != END; ++IT) {
      LoopInfo& LI = getAnalysis<LoopInfo>(*IT);
      for (LoopInfo::iterator LIT = LI.begin(), LEND = LI.end(); LIT != LEND; ++LIT) {
        handleLoop(*LIT);
      }
    }
    LLVM_DEBUG(dbgs() << "Found " << loopcounter << " loops.\n");
    return false;
  }

 private:
  int loopcounter = 0;
  void handleLoop(Loop* L) {
    ++loopcounter;
    for (Loop* SL : L->getSubLoops()) {
      handleLoop(SL);
    }
  }
};

char LoopUnroller::ID = 0;

/// Reads a module from a file.
/// On error, messages are written to stderr and null is returned.
///
/// \param Context LLVM Context for the module.
/// \param Name Input file name.
static std::unique_ptr<Module> readModule(LLVMContext& Context, StringRef Name) {
  SMDiagnostic Diag;
  std::unique_ptr<Module> Module = parseIRFile(Name, Diag, Context);

  if (!Module)
    Diag.print("llvm-canon", errs());

  return Module;
}

/// Input LLVM module file name.
cl::opt<std::string> InputFilename("f", cl::desc("Specify input filename"),
                                   cl::value_desc("filename"), cl::Required);
/// Output LLVM module file name.
cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                    cl::value_desc("filename"), cl::Required);

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              " LLVM-Unroller\n\n"
                              " This tool aims to give users fine grain control on which loops to "
                              "unroll and by which factor.\n");

  LLVMContext Context;

  std::unique_ptr<Module> Module = readModule(Context, InputFilename);

  if (!Module)
    return 1;

  LoopUnroller Canonicalizer;

  Canonicalizer.runOnModule(*Module);
  // for (auto& Function : *Module) {
  //   Canonicalizer.runOnFunction(Function);
  // }

  if (verifyModule(*Module, &errs()))
    return 1;

  std::error_code EC;
  raw_fd_ostream OutputStream(OutputFilename, EC, sys::fs::OF_None);

  if (EC) {
    errs() << EC.message();
    return 1;
  }

  Module->print(OutputStream, nullptr, false);
  OutputStream.close();
  return 0;
}

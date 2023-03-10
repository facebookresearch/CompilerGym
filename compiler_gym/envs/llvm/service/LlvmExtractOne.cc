// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//=============================================================================
// This file is a small tool to extract single functions from LLVM bitcode
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>

#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Transforms/IPO.h"

using namespace llvm;

// Command line options
static cl::OptionCategory ExtractOneOptions("llvm-extract-one Options");
static cl::opt<std::string> InputFileName(cl::Positional, cl::desc("[input file]"),
                                          cl::value_desc("filename"), cl::init("-"));
static cl::opt<std::string> OutputFileName("o",
                                           cl::desc("Specify output filename (default=stdout)"),
                                           cl::value_desc("filename"), cl::init("-"));
static cl::opt<unsigned int> Seed("seed", cl::desc("Random number seed (default=0)"),
                                  cl::value_desc("unsigned int"), cl::init(0),
                                  cl::cat(ExtractOneOptions));
static cl::opt<int> Nth("n", cl::desc("Extract the n-th function"), cl::value_desc("unsigned int"),
                        cl::init(-1), cl::cat(ExtractOneOptions));
static cl::opt<bool> CountOnly("count-only", cl::desc("Only count the number of funtions"),
                               cl::init(false), cl::cat(ExtractOneOptions));
static cl::opt<bool> ConstInits("const-inits",
                                cl::desc("Keep constant initializers (and export no functions)"),
                                cl::init(false), cl::cat(ExtractOneOptions));
static cl::opt<bool> Force("f", cl::desc("Enable binary output on terminals"),
                           cl::cat(ExtractOneOptions));
static cl::opt<bool> OutputAssembly("S", cl::desc("Write output as LLVM assembly"), cl::Hidden,
                                    cl::cat(ExtractOneOptions));
static cl::opt<bool> PreserveBitcodeUseListOrder(
    "preserve-bc-uselistorder", cl::desc("Preserve use-list order when writing LLVM bitcode."),
    cl::init(true), cl::Hidden, cl::cat(ExtractOneOptions));
static cl::opt<bool> PreserveAssemblyUseListOrder(
    "preserve-ll-uselistorder", cl::desc("Preserve use-list order when writing LLVM assembly."),
    cl::init(false), cl::Hidden, cl::cat(ExtractOneOptions));

/// Reads a module from a file.
/// On error, messages are written to stderr and null is returned.
///
/// \param Context LLVM Context for the module.
/// \param Name Input file name.
static std::unique_ptr<Module> readModule(LLVMContext& Context, StringRef Name) {
  SMDiagnostic Diag;
  std::unique_ptr<Module> Module = parseIRFile(Name, Diag, Context);

  if (!Module)
    Diag.print("llvm-extract-one", errs());

  return Module;
}

// The main tool
int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              " llvm-extract-one\n\n"
                              " Extract a single, random function from a bitcode file.\n\n"
                              " If no input file is given, or it is given as '-', then the input "
                              "file is read from stdin.\n");

  if (Seed)
    std::srand(Seed);
  else
    std::srand(std::time(nullptr));

  LLVMContext Context;

  std::unique_ptr<Module> Module = readModule(Context, InputFileName);

  if (!Module)
    return 1;

  // Find functions that might be kept
  std::vector<Function*> Functions;
  for (auto& Function : *Module) {
    if (!Function.empty()) {
      Functions.push_back(&Function);
    }
  }
  if (CountOnly) {
    std::cout << Functions.size() << std::endl;
    return 0;
  }

  // List of GlobalValues to keep
  std::vector<GlobalValue*> GVs = {};
  // Comment to put at the top of the assembly file
  std::string comment = "";

  if (ConstInits) {
    // Keep all the constants
    for (GlobalVariable& GV : Module->globals()) {
      if (GV.hasInitializer()) {
        GVs.push_back(&GV);
      }
    }
    // Set the comment
    comment = "Keep Const Inits";
  } else {
    // Work out which function to extract
    if (Functions.empty()) {
      errs() << "No suitable functions\n";
      return 1;
    }
    // Choose one
    int keeperIndex = (Nth == -1 ? std::rand() : Nth) % Functions.size();
    Function* Keeper = Functions[keeperIndex];

    // Extract the function
    ExitOnError ExitOnErr(std::string(*argv) + ": extracting function : ");
    ExitOnErr(Keeper->materialize());

    // Put it on the list
    GVs.push_back(Keeper);

    // Set the comment
    comment = "KeeperIndex = " + std::to_string(keeperIndex);
  }
  legacy::PassManager Passes;
  Passes.add(createGVExtractionPass(GVs));      // Extract the one function
  Passes.add(createStripDeadDebugInfoPass());   // Remove dead debug info
  Passes.add(createStripDeadPrototypesPass());  // Remove dead func decls

  if (verifyModule(*Module, &errs()))
    return 1;

  std::error_code EC;
  ToolOutputFile Out(OutputFileName, EC, sys::fs::OF_None);

  if (EC) {
    errs() << EC.message() << '\n';
    return 1;
  }

  if (OutputAssembly) {
    Out.os() << "; " << comment << '\n';
    Passes.add(createPrintModulePass(Out.os(), "", PreserveAssemblyUseListOrder));
  } else if (Force || !CheckBitcodeOutputToConsole(Out.os())) {
    Passes.add(createBitcodeWriterPass(Out.os(), PreserveBitcodeUseListOrder));
  }
  Passes.run(*Module);

  // Declare success.
  Out.keep();

  return 0;
}

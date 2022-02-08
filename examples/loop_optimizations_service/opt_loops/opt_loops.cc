// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "compiler_gym/third_party/LLVM-Canon/IRCanonicalizer.h"
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
#include "nlohmann/json.hpp"

using namespace llvm;

using namespace llvm::yaml;
using llvm::yaml::IO;
using llvm::yaml::Output;
using llvm::yaml::ScalarEnumerationTraits;
using json = nlohmann::json;

// a version of LLVM's getOptionalIntLoopAttribute(...) that accepts `const` Loop as argument
// this is required to invoke in to_json(...)
llvm::Optional<int> getOptionalIntLoopAttribute1(const Loop* TheLoop, StringRef Name) {
  const MDOperand* AttrMD = findStringMetadataForLoop(TheLoop, Name).getValueOr(nullptr);
  if (!AttrMD)
    return None;

  ConstantInt* IntMD = mdconst::extract_or_null<ConstantInt>(AttrMD->get());
  if (!IntMD)
    return None;

  return IntMD->getSExtValue();
}

static Optional<bool> getOptionalBoolLoopAttribute(const Loop* TheLoop, StringRef Name) {
  MDNode* MD = findOptionMDForLoop(TheLoop, Name);
  if (!MD)
    return None;
  switch (MD->getNumOperands()) {
    case 1:
      // When the value is absent it is interpreted as 'attribute set'.
      return true;
    case 2:
      if (ConstantInt* IntMD = mdconst::extract_or_null<ConstantInt>(MD->getOperand(1).get()))
        return IntMD->getZExtValue();
      return true;
  }
  llvm_unreachable("unexpected number of options");
}

static bool getBooleanLoopAttribute(const Loop* TheLoop, StringRef Name) {
  return getOptionalBoolLoopAttribute(TheLoop, Name).getValueOr(false);
}

std::string getStringMetadataFromLoop(Loop*& L, const char* MDString) {
  Optional<const MDOperand*> Value = findStringMetadataForLoop(L, MDString);
  if (!Value)
    return "None";

  const MDOperand* Op = *Value;
  assert(Op && mdconst::hasa<ConstantInt>(*Op) && "invalid metadata");
  return std::to_string(mdconst::extract<ConstantInt>(*Op)->getZExtValue());
}

struct LoopConfig {
  std::string FName;
  std::string MName;
  std::string IDStr;
  unsigned int MetadataID;
  std::string Name;
  int Depth;
  std::string HeaderName;
  bool MetaLoopUnrollEnable;
  bool MetaLoopUnrollDisable;
  llvm::Optional<int> MetaLoopUnrollCount;
  bool MetaLoopIsUnrolled;
  bool MetaLoopVectorEnable;
  bool MetaLoopVectorDisable;
  llvm::Optional<int> MetaLoopVectorWidth;
  bool MetaLoopIsVectorized;
  std::string IR;

  LoopConfig() {}

  LoopConfig(Loop*& L) {
    Function* F = L->getBlocks()[0]->getParent();
    FName = F->getName();

    Module* M = F->getParent();
    MName = M->getName();

    llvm::raw_string_ostream IDStream(IDStr);
    L->getLoopID()->printAsOperand(IDStream, M);

    MetadataID = L->getLoopID()->getMetadataID();  // this id always prints a value of 4. Not sure
                                                   // if I am using it correctly
    Name = L->getName();  // NOTE: actually L->getName calls L->getHeader()->getName()
    Depth = L->getLoopDepth();

    // TODO: find a way to provide a Name to the loop that will remain consisten across multiple
    // `opt` calls
    HeaderName = L->getHeader()->getName();
    static int Count = 0;
    if (HeaderName.length() == 0) {
      HeaderName = "loop_" + std::to_string(Count++);
      L->getHeader()->setName(HeaderName);
    }

    MetaLoopUnrollEnable = getBooleanLoopAttribute(L, "llvm.loop.unroll.enable");
    MetaLoopUnrollDisable = getBooleanLoopAttribute(L, "llvm.loop.unroll.disable");
    MetaLoopUnrollCount = getOptionalIntLoopAttribute(L, "llvm.loop.unroll.count");
    MetaLoopIsUnrolled = getBooleanLoopAttribute(L, "llvm.loop.isunrolled");
    MetaLoopVectorEnable = getBooleanLoopAttribute(L, "llvm.loop.vector.enable");
    MetaLoopVectorDisable = getBooleanLoopAttribute(L, "llvm.loop.vector.disable");
    MetaLoopVectorWidth = getOptionalIntLoopAttribute(L, "llvm.loop.vector.width");
    MetaLoopIsVectorized = getBooleanLoopAttribute(L, "llvm.loop.isvectorized");

    // dump the IR of the loop
    llvm::raw_string_ostream stream(IR);
    L->print(stream, true, true);
  }
};

void to_json(json& j, const LoopConfig& LC) {
  j["ID"] = LC.IDStr;
  j["Function"] = LC.FName;
  j["Module"] = LC.MName;
  j["MetadataID"] = LC.MetadataID;
  j["Name"] = LC.Name;
  j["Depth"] = LC.Depth;
  j["HeaderName"] = LC.HeaderName;
  j["llvm.loop.unroll.enable"] = LC.MetaLoopUnrollEnable;
  j["llvm.loop.unroll.disable"] = LC.MetaLoopUnrollDisable;
  if (LC.MetaLoopUnrollCount.hasValue())
    j["llvm.loop.unroll.count"] = LC.MetaLoopUnrollCount.getValue();
  j["llvm.loop.isunrolled"] = LC.MetaLoopIsUnrolled;
  j["llvm.loop.vectorize.enable"] = LC.MetaLoopVectorEnable;
  j["llvm.loop.vectorize.disable"] = LC.MetaLoopVectorDisable;
  if (LC.MetaLoopVectorWidth.hasValue())
    j["llvm.loop.vectorize.width"] = LC.MetaLoopVectorWidth.getValue();
  j["llvm.loop.isvectorized"] = LC.MetaLoopIsVectorized;
  j["llvm"] = LC.IR;
}

template <>
struct llvm::yaml::MappingTraits<LoopConfig> {
  static void mapping(IO& io, LoopConfig& LC) {
    io.mapRequired("ID", LC.IDStr);
    io.mapRequired("Function", LC.FName);
    io.mapRequired("Module", LC.MName);
    io.mapRequired("MetadataID", LC.MetadataID);
    io.mapRequired("Name", LC.Name);
    io.mapRequired("Depth", LC.Depth);
    io.mapRequired("HeaderName", LC.HeaderName);
    io.mapOptional("llvm.loop.unroll.enable", LC.MetaLoopUnrollEnable);
    io.mapOptional("llvm.loop.unroll.disable", LC.MetaLoopUnrollDisable);
    io.mapOptional("llvm.loop.unroll.count", LC.MetaLoopUnrollCount);
    io.mapOptional("llvm.loop.isunrolled", LC.MetaLoopIsUnrolled);
    io.mapOptional("llvm.loop.vectorize.enable", LC.MetaLoopVectorEnable);
    io.mapOptional("llvm.loop.vectorize.disable", LC.MetaLoopVectorDisable);
    io.mapOptional("llvm.loop.vectorize.width", LC.MetaLoopVectorWidth);
    io.mapOptional("llvm.loop.isvectorized", LC.MetaLoopIsVectorized);
    io.mapOptional("llvm", LC.IR);
  }
};

namespace {
/// Input LLVM module file name.
cl::opt<std::string> InputFilename(cl::Positional, cl::desc("Specify input filename"),
                                   cl::value_desc("filename"), cl::init("-"));
/// Output LLVM module file name.
cl::opt<std::string> OutputFilename("o", cl::desc("Specify output filename"),
                                    cl::value_desc("filename"), cl::init("-"));

/// Output Loop Configuration/Features file in various formats.
cl::opt<std::string> OutputYAMLFile("emit-yaml", cl::desc("Specify output YAML log filename"),
                                    cl::value_desc("filename"), cl::init("/tmp/loops.yaml"));

cl::opt<std::string> OutputJSONFile("emit-json", cl::desc("Specify output JSON log filename"),
                                    cl::value_desc("filename"), cl::init("/tmp/loops.json"));

// TODO(mostafaelhoushi): add other features like "read-yaml", "print-yaml-after-all",
// "print-yaml-before-all", "print-yaml-after=<list of passes>", "print-yaml-before=<list of
// passes>" etc.

/// Loop Optimizations
static cl::opt<bool> UnrollEnable("floop-unroll", cl::desc("Enable loop unrolling"),
                                  cl::init(false));

static cl::opt<unsigned> UnrollCount(
    "funroll-count", cl::desc("Use this unroll count for all loops including those with "
                              "unroll_count pragma values, for testing purposes"));

static cl::opt<bool> VectorizeEnable("floop-vectorize", cl::desc("Enable loop vectorize"),
                                     cl::init(false));

static cl::opt<unsigned> VectorizationFactor("fforce-vector-width",
                                             cl::desc("Sets the SIMD width. Zero is autoselect."));

// Force binary on terminals
static cl::opt<bool> Force("f", cl::desc("Enable binary output on terminals"));

// Output assembly
static cl::opt<bool> OutputAssembly("S", cl::desc("Write output as LLVM assembly"));

// Canonicalize the IR
static cl::opt<bool> Canonicalize("canonicalize", cl::desc("Canonicalize the IR"), cl::init(false));

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

std::vector<LoopConfig> LCs;
class LoopLog : public llvm::FunctionPass {
 public:
  static char ID;
  std::unordered_map<std::string, int> Counts;

  LoopLog() : FunctionPass(ID) {}

  virtual void getAnalysisUsage(AnalysisUsage& AU) const override {
    AU.addRequired<LoopInfoWrapperPass>();
  }

  bool runOnFunction(llvm::Function& F) override {
    LoopInfo& LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
    auto Loops = LI.getLoopsInPreorder();

    // Should really account for module, too.
    Counts[F.getName().str()] = Loops.size();

    // ensure that all loops have metadata
    for (auto L : Loops) {
      if (L->getLoopID() == nullptr) {
        // workaround to add metadata
        // TODO: is there a better way to add metadata to aloop?
        addStringMetadataToLoop(L, "custom.label.loop", true);
      }
    }

    for (auto L : Loops) {
      LoopConfig LC(L);
      LCs.push_back(LC);
    }

    return false;
  }
};

char LoopLog::ID = 0;

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
/// \param Name Input file Name.
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
void initializeLoopLogPass(PassRegistry& Registry);
void initializeLoopConfiguratorPassPass(PassRegistry& Registry);
}  // namespace llvm

// Initialise the pass. We have to declare the dependencies we use.
INITIALIZE_PASS_BEGIN(LoopLog, "log-loops", "Log loops IR and configuration", false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopLog, "log-loops", "Log loops IR and configuration", false, false)

INITIALIZE_PASS_BEGIN(LoopConfiguratorPass, "loops-configurator", "Configurates loop optimization",
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(LoopConfiguratorPass, "loops-configurator", "Configurates loop optimization",
                    false, false)

namespace llvm {
Pass* createLoopLogPass() { return new LoopLog(); }
}  // end namespace llvm

LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(LoopConfig)

int main(int argc, char** argv) {
  cl::ParseCommandLineOptions(argc, argv,
                              " opt_loops\n\n"
                              " Fine grain loop optimizer and configuration logger.\n");

  LLVMContext Context;
  SMDiagnostic Err;
  SourceMgr SM;
  std::error_code EC;

  std::unique_ptr<Module> Module = readModule(Context, InputFilename);

  if (!Module)
    return 1;

  // Prepare output IR file
  ToolOutputFile Out(OutputFilename, EC, sys::fs::OF_None);
  if (EC) {
    Err = SMDiagnostic(OutputFilename, SourceMgr::DK_Error,
                       "Could not open output file: " + EC.message());
    Err.print(argv[0], errs());
    return 1;
  }

  // Prepare loops dump/configuration yaml file
  raw_fd_ostream ToolYAMLFile(OutputYAMLFile, EC);
  yaml::Output Yaml(ToolYAMLFile);

  // Prepare loops dump/configuration json file
  std::fstream ToolJSONFile(OutputJSONFile, std::fstream::out);

  initializeLoopLogPass(*PassRegistry::getPassRegistry());
  OptCustomPassManager PM;

  // Canonicalize IR
  if (Canonicalize) {
    IRCanonicalizer* Canonicalizer =
        new IRCanonicalizer(PreserveOrder, RenameAll, FoldPreoutputs, ReorderOperands);

    PM.add(Canonicalizer);
  }
  LoopConfiguratorPass* LoopConfigurator = new LoopConfiguratorPass();
  PM.add(LoopConfigurator);
  PM.add(createLoopUnrollPass());
  PM.add(createLICMPass());
  PM.add(createLoopVectorizePass(false, false));
  PassManagerBuilder Builder;
  Builder.LoopVectorize = VectorizeEnable;
  Builder.populateModulePassManager(PM);

  PM.add(createLoopLogPass());

  // PM to output the module
  if (OutputAssembly) {
    PM.add(createPrintModulePass(Out.os(), "", PreserveAssemblyUseListOrder));
  } else if (Force || !CheckBitcodeOutputToConsole(Out.os())) {
    PM.add(createBitcodeWriterPass(Out.os(), PreserveBitcodeUseListOrder));
  }
  PM.run(*Module);
  Out.keep();

  // Log loop configuration
  auto jsonObjects = json::array();
  jsonObjects = LCs;  // this invokes to_json(json& j, const LoopConfig& LC)
  ToolJSONFile << jsonObjects;

  Yaml << LCs;  // this invokes mapping(IO& io, LoopConfig& LC) and writes to file
  ToolYAMLFile.close();

  return 0;
}

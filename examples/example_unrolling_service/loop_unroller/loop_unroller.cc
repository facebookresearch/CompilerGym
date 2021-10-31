//==============================================================================
// Estimate best and worst case execution time of LLVM code.
//
// Hugh Leather hughleat@gmail.com 2020-06-30
//==============================================================================

#include <cassert>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/CFG.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace std;

//------------------------------------------------------------------------------
// Command line options
//------------------------------------------------------------------------------
cl::OptionCategory bwcetCategory{"bwcet options"};

cl::list<std::string> inputFiles(cl::Positional, cl::desc{"<Modules to analyse>"},
                                 cl::value_desc{"bitcode filename"}, cl::OneOrMore,
                                 cl::cat{bwcetCategory});

cl::opt<string> outputFilename("output", cl::desc("Specify output filename (default to std out)"),
                               cl::value_desc("output filename"), cl::init("-"),
                               cl::cat{bwcetCategory});
cl::alias outputFilenameA("o", cl::desc("Alias for --output"), cl::aliasopt(outputFilename),
                          cl::cat{bwcetCategory});

enum OutputFormat { TXT, JSON, CSV };
cl::opt<OutputFormat> outputFormat("format", cl::desc("Choose output format"),
                                   cl::values(clEnumVal(TXT, "Human readable format (default)"),
                                              clEnumVal(JSON, "JSON format"),
                                              clEnumVal(CSV, "CSV format")),
                                   cl::init(TXT), cl::cat{bwcetCategory});
cl::alias outputFormatA("f", cl::desc("Alias for --format"), cl::aliasopt(outputFormat),
                        cl::cat{bwcetCategory});

cl::opt<TargetTransformInfo::TargetCostKind> costKind(
    "cost-kind", cl::desc("Target cost kind"), cl::init(TargetTransformInfo::TCK_RecipThroughput),
    cl::values(clEnumValN(TargetTransformInfo::TCK_RecipThroughput, "throughput",
                          "Reciprocal throughput (default)"),
               clEnumValN(TargetTransformInfo::TCK_Latency, "latency", "Instruction latency"),
               clEnumValN(TargetTransformInfo::TCK_CodeSize, "code-size", "Code size")),
    cl::cat{bwcetCategory});
cl::alias costKindA("k", cl::desc("Alias for --cost-kind"), cl::aliasopt(costKind),
                    cl::cat{bwcetCategory});

//------------------------------------------------------------------------------
// Determine if CFG is a DAG.
//------------------------------------------------------------------------------
// Colour for DFS
enum Colour { WHITE, GREY, BLACK };
// DFS
bool isDAG(const BasicBlock* bb, unordered_map<const BasicBlock*, Colour>& colour) {
  switch (colour[bb]) {
    case BLACK:
      return true;
    case GREY:
      return false;
    case WHITE: {
      colour[bb] = GREY;
      for (const auto* succ : successors(bb)) {
        if (!isDAG(succ, colour))
          return false;
      }
      colour[bb] = BLACK;
      return true;
    }
  }
}
bool isDAG(const Function& f) {
  unordered_map<const BasicBlock*, Colour> colour;
  return isDAG(&f.getEntryBlock(), colour);
}

//------------------------------------------------------------------------------
// Get min and max cost of functions and basic blocks
//------------------------------------------------------------------------------
TargetIRAnalysis tira;
unique_ptr<TargetTransformInfoWrapperPass> ttiwp((TargetTransformInfoWrapperPass*)
                                                     createTargetTransformInfoWrapperPass(tira));
using CostT = double;

CostT getCost(const BasicBlock& bb, const TargetTransformInfo& tti) {
  CostT cost = 0;
  for (const auto& insn : bb) {
    cost += tti.getInstructionCost(&insn, costKind);
  }
  return cost;
}
CostT minCost(const Function& f, unordered_map<const BasicBlock*, CostT> bbCost) {
  // Cost of best path (path with minimum cost)
  CostT best = numeric_limits<CostT>::infinity();
  // The exit block with the best path cost
  const BasicBlock* bestBB = nullptr;
  // Predecessors
  unordered_map<const BasicBlock*, const BasicBlock*> pred;
  // Map of costs into each vertex
  unordered_map<const BasicBlock*, CostT> costIn;
  // Priority queue
  set<pair<CostT, const BasicBlock*>> q;
  // Pointers into q - so we can change priority
  unordered_map<const BasicBlock*, decltype(q.begin())> iter;
  // Initialise cost
  for (const BasicBlock& v : f.getBasicBlockList()) costIn[&v] = numeric_limits<CostT>::infinity();
  auto start = &f.getEntryBlock();
  costIn[start] = 0;
  // Push into q (and remember iterator)
  auto iti = q.insert({costIn[start], start});
  iter[start] = iti.first;
  // Do the search
  while (!q.empty()) {
    // Pop from the q
    auto top = q.begin();
    const BasicBlock* v = top->second;
    CostT cIn = top->first;
    q.erase(top);
    iter.erase(v);
    assert(cIn == costIn[v]);

    // Get the cost out of this node
    int cOut = cIn + bbCost[v];
    // Count the successors as we process them
    int numSuccs = 0;
    // Process each successor
    for (const auto* succ : successors(v)) {
      numSuccs++;
      // Update if the cost is better
      if (cOut < costIn[succ]) {
        // Set the new cost
        costIn[succ] = cOut;
        // Delete from the queue if already in there
        if (iter.count(succ)) {
          auto it = iter[succ];
          q.erase(it);
        }
        // Insert into the queue (and remember iterator)
        auto iti = q.insert({cOut, succ});
        iter[succ] = iti.first;
        // Remember predecessor
        pred[succ] = v;
      }
    }
    // Update best if this is an exit block (no successors) and we have a better cost
    if (numSuccs == 0 && best > cOut) {
      best = cOut;
      bestBB = v;
    }
  }
  return best;
}
CostT maxCost(const Function& f, unordered_map<const BasicBlock*, CostT> bbCost) {
  // Cost of best path (path with minimum cost)
  CostT best = 0;
  // The exit block with the best path cost
  const BasicBlock* bestBB = nullptr;
  // Predecessors
  unordered_map<const BasicBlock*, const BasicBlock*> pred;
  // Map of costs into each vertex
  unordered_map<const BasicBlock*, CostT> costIn;
  // Priority queue
  struct RCmp {
    bool operator()(const pair<CostT, const BasicBlock*>& a,
                    const pair<CostT, const BasicBlock*>& b) const {
      if (a.first == b.first)
        return a.second < b.second;
      return a.first > b.first;
    }
  };
  set<pair<CostT, const BasicBlock*>, RCmp> q;
  // Pointers into q - so we can change priority
  unordered_map<const BasicBlock*, decltype(q.begin())> iter;
  // Initialise cost
  for (const BasicBlock& v : f.getBasicBlockList()) costIn[&v] = 0;
  auto start = &f.getEntryBlock();
  costIn[start] = 0;
  // Push into q (and remember iterator)
  auto iti = q.insert({costIn[start], start});
  iter[start] = iti.first;
  // Do the search
  while (!q.empty()) {
    // Pop from the q
    auto top = q.begin();
    const BasicBlock* v = top->second;
    CostT cIn = top->first;
    q.erase(top);
    iter.erase(v);
    assert(cIn == costIn[v]);

    // Get the cost out of this node
    int cOut = cIn + bbCost[v];
    // Count the successors as we process them
    int numSuccs = 0;
    // Process each successor
    for (const auto* succ : successors(v)) {
      numSuccs++;
      // Update if the cost is better
      if (cOut > costIn[succ]) {
        // Set the new cost
        costIn[succ] = cOut;
        // Delete from the queue if already in there
        if (iter.count(succ)) {
          auto it = iter[succ];
          q.erase(it);
        }
        // Insert into the queue (and remember iterator)
        auto iti = q.insert({cOut, succ});
        iter[succ] = iti.first;
        // Remember predecessor
        pred[succ] = v;
      }
    }
    // Update best if this is an exit block (no successors) and we have a better cost
    if (numSuccs == 0 && best < cOut) {
      best = cOut;
      bestBB = v;
    }
  }
  return best;
}

pair<CostT, CostT> getCost(const Function& f) {
  auto& tti = ttiwp->getTTI(f);

  // Precompute BB costs.
  unordered_map<const BasicBlock*, CostT> bbCost;
  for (const auto& bb : f.getBasicBlockList()) bbCost[&bb] = getCost(bb, tti);

  if (isDAG(f)) {
    return {minCost(f, bbCost), maxCost(f, bbCost)};
  } else {
    return {minCost(f, bbCost), numeric_limits<CostT>::infinity()};
  }
}

//------------------------------------------------------------------------------
// Visitor functions, called to process the module
//------------------------------------------------------------------------------
void visit(const Function& f, ostream& os) {
  auto costs = getCost(f);
  switch (outputFormat) {
    case TXT: {
      os << "  Function: " << f.getName().str() << " ";
      os << "min=" << costs.first << " ";
      os << "max=" << costs.second << endl;
      break;
    }
    case JSON: {
      os << "{";
      os << "\"function\":\"" << f.getName().str() << "\",";
      os << "\"min\":" << costs.first;
      if (costs.second != numeric_limits<CostT>::infinity()) {
        os << ",\"max\":" << costs.second;
      }
      os << "}";
      break;
    }
    case CSV: {
      os << f.getParent()->getName().str() << ",";
      os << f.getName().str() << ",";
      os << costs.first << ",";
      if (costs.second != numeric_limits<CostT>::infinity()) {
        os << costs.second;
      }
      os << "\n";
      break;
    }
  }
}
void visit(const Module& m, ostream& os) {
  switch (outputFormat) {
    case TXT: {
      os << "Module: " << m.getName().str() << "\n";
      for (const auto& f : m.functions()) visit(f, os);
      break;
    }
    case JSON: {
      os << "{";
      os << "\"module\":\"" << m.getName().str() << "\",";
      os << "\"functions\":[";
      bool isFirst = true;
      for (const auto& f : m.functions()) {
        if (!isFirst)
          os << ",";
        else
          isFirst = false;
        visit(f, os);
      }
      os << "]}";
      break;
    }
    case CSV: {
      for (const auto& f : m.functions()) visit(f, os);
      break;
    }
  }
}
void visit(const string& filename, ostream& os) {
  // Parse the IR file passed on the command line.
  SMDiagnostic err;
  LLVMContext ctx;
  unique_ptr<Module> m = parseIRFile(filename, err, ctx);

  if (!m)
    throw err;

  // Run the analysis and print the results
  visit(*m, os);
}
void visit(const vector<string>& filenames, ostream& os) {
  switch (outputFormat) {
    case TXT: {
      for (const auto& fn : filenames) visit(fn, os);
      break;
    }
    case JSON: {
      os << "[";
      bool isFirst = true;
      for (const auto& fn : filenames) {
        if (!isFirst)
          os << ",";
        else
          isFirst = false;
        visit(fn, os);
      }
      os << "]\n";
      break;
    }
    case CSV: {
      os << "Module, Function, DAG, Min, Max\n";
      for (const auto& fn : filenames) visit(fn, os);
      break;
    }
  }
}
//------------------------------------------------------------------------------
// Driver
//------------------------------------------------------------------------------
int main(int argc, char** argv) {
  // Hide all options apart from the ones specific to this tool
  cl::HideUnrelatedOptions(bwcetCategory);

  cl::ParseCommandLineOptions(
      argc, argv,
      "Estimates the best and worst case runtime for each function the input IR file\n");

  try {
    // Get the output file
    unique_ptr<ostream> ofs(outputFilename == "-" ? nullptr : new ofstream(outputFilename.c_str()));
    if (ofs && !ofs->good()) {
      throw "Error opening output file: " + outputFilename;
    }
    ostream& os = ofs ? *ofs : cout;

    // Makes sure llvm_shutdown() is called (which cleans up LLVM objects)
    // http://llvm.org/docs/ProgrammersManual.html#ending-execution-with-llvm-shutdown
    llvm_shutdown_obj shutdown_obj;

    // Do the work
    visit(inputFiles, os);

  } catch (string e) {
    errs() << e;
    return -1;
  } catch (SMDiagnostic e) {
    e.print(argv[0], errs(), false);
    return -1;
  }
  return 0;
}

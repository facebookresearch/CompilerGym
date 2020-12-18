// A modified version of the LLVM InstCount analysis pass which produces the
// features used in the work:
//
//   Huang, Q., Haj-Ali, A., Moses, W., Xiang, J., Stoica, I., Asanovic, K., &
//   Wawrzynek, J. (2019). Autophase: Compiler phase-ordering for hls with deep
//   reinforcement learning. FCCM. https://doi.org/10.1109/FCCM.2019.00049

#pragma once

#include <vector>

#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Pass.h"

// NOTE(cummins): I would prefer not to pull in the LLVM namespace here but this
// is required for the Instruction.def macros.
using namespace llvm;

namespace autophase {

constexpr size_t kAutophaseFeatureDimensionality = 56;

class InstCount : public FunctionPass, public InstVisitor<InstCount> {
 public:
  // Get the counter values as a vector of integers.
  static std::vector<int64_t> getFeatureVector(llvm::Module&);

 private:
  InstCount() : FunctionPass(ID) {}
  friend class InstVisitor<InstCount>;

  bool runOnFunction(Function& F) override;

  // Declare a counter variable and a getter function.
#define COUNTER(name, unused_description) \
  int64_t name = 0;                       \
  int64_t get_##name() const { return name; }

  // Custom autophase counters.
  COUNTER(TotalInsts, "Number of instructions (of all types)");
  COUNTER(TotalBlocks, "Number of basic blocks");
  COUNTER(BlockLow, "Number of BB's with less than 15 instructions");
  COUNTER(BlockMid, "Number of BB's with instructions between [15, 500]");
  COUNTER(BlockHigh, "Number of BB's with more than 500 instructions");
  COUNTER(TotalFuncs, "Number of non-external functions");
  COUNTER(TotalMemInst, "Number of memory instructions");
  COUNTER(BeginPhi, "# of Phi-nodes at beginning of BB");
  COUNTER(ArgsPhi, "Total arguments to Phi nodes");
  COUNTER(BBNoPhi, "# of BB's with no Phi nodes");
  COUNTER(BB03Phi, "# of BB's with Phi node # in range (0, 3]");
  COUNTER(BBHiPhi, "# of BB's with more than 3 Phi nodes");
  COUNTER(BBNumArgsHi, "# of BB where total args for phi nodes > 5");
  COUNTER(BBNumArgsLo, "# of BB where total args for phi nodes is [1, 5]");
  COUNTER(testUnary, "Unary");
  COUNTER(binaryConstArg, "Binary operations with a constant operand");
  COUNTER(callLargeNumArgs, "# of calls with number of arguments > 4");
  COUNTER(returnInt, "# of calls that return an int");
  COUNTER(oneSuccessor, "# of BB's with 1 successor");
  COUNTER(twoSuccessor, "# of BB's with 2 successors");
  COUNTER(moreSuccessors, "# of BB's with >2 successors");
  COUNTER(onePred, "# of BB's with 1 predecessor");
  COUNTER(twoPred, "# of BB's with 2 predecessors");
  COUNTER(morePreds, "# of BB's with >2 predecessors");
  COUNTER(onePredOneSuc, "# of BB's with 1 predecessor and 1 successor");
  COUNTER(onePredTwoSuc, "# of BB's with 1 predecessor and 2 successors");
  COUNTER(twoPredOneSuc, "# of BB's with 2 predecessors and 1 successor");
  COUNTER(twoEach, "# of BB's with 2 predecessors and successors");
  COUNTER(moreEach, "# of BB's with >2 predecessors and successors");
  COUNTER(NumEdges, "# of edges");
  COUNTER(CriticalCount, "# of critical edges");
  COUNTER(BranchCount, "# of branches");
  COUNTER(numConstOnes, "# of occurrences of constant 1");
  COUNTER(numConstZeroes, "# of occurrences of constant 0");
  COUNTER(const32Bit, "# of occurrences of 32-bit integer constants");
  COUNTER(const64Bit, "# of occurrences of 64-bit integer constants");
  COUNTER(UncondBranches, "# of unconditional branches");

// Generate opcode counters.
#define HANDLE_INST(N, OPCODE, CLASS) COUNTER(Num##OPCODE##Inst, "Number of " #OPCODE " insts");

#include "llvm/IR/Instruction.def"

  static char ID;

  void visitFunction(Function& F);

  void visitBasicBlock(BasicBlock& BB);

// Generate instruction visitors.
#define HANDLE_INST(N, OPCODE, CLASS) void visit##OPCODE(CLASS&);

#include "llvm/IR/Instruction.def"

  void visitInstruction(Instruction& I);
};

}  // namespace autophase

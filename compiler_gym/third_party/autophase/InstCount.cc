//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//

#include "compiler_gym/third_party/autophase/InstCount.h"

#include "llvm/Analysis/CFG.h"
#include "llvm/Support/raw_ostream.h"

namespace autophase {

char InstCount::ID = 0;

bool InstCount::runOnFunction(Function& F) {
  unsigned StartMemInsts = NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
                           NumInvokeInst + NumAllocaInst;
  visit(F);
  unsigned EndMemInsts = NumGetElementPtrInst + NumLoadInst + NumStoreInst + NumCallInst +
                         NumInvokeInst + NumAllocaInst;
  TotalMemInst += EndMemInsts - StartMemInsts;
  return false;
}

void InstCount::visitFunction(Function& F) { ++TotalFuncs; }

void InstCount::visitBasicBlock(BasicBlock& BB) {
  ++TotalBlocks;
  Instruction* term = BB.getTerminator();
  unsigned numSuccessors = term->getNumSuccessors();
  for (int i = 0; i < numSuccessors; i++) {
    NumEdges++;
    if (isCriticalEdge(term, i)) {
      CriticalCount++;
    }
  }
  unsigned numPreds = 0;
  for (pred_iterator pi = pred_begin(&BB), E = pred_end(&BB); pi != E; ++pi) {
    numPreds++;
  }
  if (numSuccessors == 1) {
    oneSuccessor++;
  } else if (numSuccessors == 2) {
    twoSuccessor++;

  } else if (numSuccessors > 2) {
    moreSuccessors++;
  }
  if (numPreds == 1) {
    onePred++;
  } else if (numPreds == 2) {
    twoPred++;
  } else if (numPreds > 2) {
    morePreds++;
  }

  if (numPreds == 1 && numSuccessors == 1) {
    onePredOneSuc++;
  } else if (numPreds == 2 && numSuccessors == 1) {
    twoPredOneSuc++;
  } else if (numPreds == 1 && numSuccessors == 2) {
    onePredTwoSuc++;
  } else if (numPreds == 2 && numSuccessors == 2) {
    twoEach++;
  } else if (numPreds > 2 && numSuccessors > 2) {
    moreEach++;
  }

  unsigned tempCount = 0;
  bool isFirst = true;
  unsigned phiCount = 0;
  unsigned BBArgs = 0;
  for (Instruction& I : BB) {
    if (auto* bi = dyn_cast<BranchInst>(&I)) {
      BranchCount++;
      if (bi->isUnconditional()) {
        UncondBranches++;
      }
    }
    for (int i = 0; i < I.getNumOperands(); i++) {
      Value* v = I.getOperand(i);
      // Type* t = v->getType();
      if (auto* c = dyn_cast<Constant>(v)) {
        if (auto* ci = dyn_cast<ConstantInt>(c)) {
          APInt val = ci->getValue();
          unsigned bitWidth = val.getBitWidth();
          if (bitWidth == 32) {
            const32Bit++;
          } else if (bitWidth == 64) {
            const64Bit++;
          }
          if (val == 1) {
            numConstOnes++;
          } else if (val == 0) {
            numConstZeroes++;
          }
        }
      }
    }
    if (isa<CallInst>(I)) {
      if (cast<CallInst>(I).getNumArgOperands() > 4) {
        callLargeNumArgs++;
      }
      auto calledFunction = cast<CallInst>(I).getCalledFunction();
      if (calledFunction) {
        auto returnType = calledFunction->getReturnType();
        if (returnType) {
          if (returnType->isIntegerTy()) {
            returnInt++;
          }
        }
      }
    }
    if (isa<UnaryInstruction>(I)) {
      testUnary++;
    }
    if (isa<BinaryOperator>(I)) {
      if (isa<Constant>(I.getOperand(0)) || isa<Constant>(I.getOperand(1))) {
        binaryConstArg++;
      }
    }
    if (isFirst && isa<PHINode>(I)) {
      BeginPhi++;
    }
    if (isa<PHINode>(I)) {
      phiCount++;
      unsigned inc = cast<PHINode>(I).getNumIncomingValues();
      ArgsPhi += inc;
      BBArgs += inc;
    }
    isFirst = false;
    tempCount++;
  }
  if (phiCount == 0) {
    BBNoPhi++;
  } else if (phiCount <= 3) {
    BB03Phi++;
  } else {
    BBHiPhi++;
  }
  if (BBArgs > 5) {
    BBNumArgsHi++;
  } else if (BBArgs >= 1) {
    BBNumArgsLo++;
  }
  if (tempCount < 15) {
    BlockLow++;
  } else if (tempCount <= 500) {
    BlockMid++;
  } else {
    BlockHigh++;
  }
}

// Generate instruction visitors.
#define HANDLE_INST(N, OPCODE, CLASS)     \
  void InstCount::visit##OPCODE(CLASS&) { \
    ++Num##OPCODE##Inst;                  \
    ++TotalInsts;                         \
  }

#include "llvm/IR/Instruction.def"

void InstCount::visitInstruction(Instruction& I) {
  errs() << "Instruction Count does not know about " << I;
  llvm_unreachable(nullptr);
}

std::vector<int64_t> InstCount::getFeatureVector(Module& module) {
  InstCount pass;
  for (auto& function : module) {
    pass.runOnFunction(function);
  }

  std::vector<int64_t> features;
  features.reserve(kAutophaseFeatureDimensionality);
  features.push_back(pass.get_BBNumArgsHi());
  features.push_back(pass.get_BBNumArgsLo());
  features.push_back(pass.get_onePred());
  features.push_back(pass.get_onePredOneSuc());
  features.push_back(pass.get_onePredTwoSuc());
  features.push_back(pass.get_oneSuccessor());
  features.push_back(pass.get_twoPred());
  features.push_back(pass.get_twoPredOneSuc());
  features.push_back(pass.get_twoEach());
  features.push_back(pass.get_twoSuccessor());
  features.push_back(pass.get_morePreds());
  features.push_back(pass.get_BB03Phi());
  features.push_back(pass.get_BBHiPhi());
  features.push_back(pass.get_BBNoPhi());
  features.push_back(pass.get_BeginPhi());
  features.push_back(pass.get_BranchCount());
  features.push_back(pass.get_returnInt());
  features.push_back(pass.get_CriticalCount());
  features.push_back(pass.get_NumEdges());
  features.push_back(pass.get_const32Bit());
  features.push_back(pass.get_const64Bit());
  features.push_back(pass.get_numConstZeroes());
  features.push_back(pass.get_numConstOnes());
  features.push_back(pass.get_UncondBranches());
  features.push_back(pass.get_binaryConstArg());
  features.push_back(pass.get_NumAShrInst());
  features.push_back(pass.get_NumAddInst());
  features.push_back(pass.get_NumAllocaInst());
  features.push_back(pass.get_NumAndInst());
  features.push_back(pass.get_BlockMid());
  features.push_back(pass.get_BlockLow());
  features.push_back(pass.get_NumBitCastInst());
  features.push_back(pass.get_NumBrInst());
  features.push_back(pass.get_NumCallInst());
  features.push_back(pass.get_NumGetElementPtrInst());
  features.push_back(pass.get_NumICmpInst());
  features.push_back(pass.get_NumLShrInst());
  features.push_back(pass.get_NumLoadInst());
  features.push_back(pass.get_NumMulInst());
  features.push_back(pass.get_NumOrInst());
  features.push_back(pass.get_NumPHIInst());
  features.push_back(pass.get_NumRetInst());
  features.push_back(pass.get_NumSExtInst());
  features.push_back(pass.get_NumSelectInst());
  features.push_back(pass.get_NumShlInst());
  features.push_back(pass.get_NumStoreInst());
  features.push_back(pass.get_NumSubInst());
  features.push_back(pass.get_NumTruncInst());
  features.push_back(pass.get_NumXorInst());
  features.push_back(pass.get_NumZExtInst());
  features.push_back(pass.get_TotalBlocks());
  features.push_back(pass.get_TotalInsts());
  features.push_back(pass.get_TotalMemInst());
  features.push_back(pass.get_TotalFuncs());
  features.push_back(pass.get_ArgsPhi());
  features.push_back(pass.get_testUnary());
  return features;
}

}  // namespace autophase

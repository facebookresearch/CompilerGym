// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Expose the LLVM -instcount pass as a feature extractor.
//
// Based on thethe InstCount pass:
//
//===-- InstCount.cpp - Collects the count of all instructions ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass collects the count of all instructions and reports them
//
//===----------------------------------------------------------------------===//
#pragma once

#include <array>
#include <string>

#include "llvm/Analysis/Passes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/Pass.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

namespace compiler_gym::llvm_service {

// Determine the dimensionality of the InstCount feature vector at compile time.
// We have 3 predefined counters (TotalInsts, TotalBlocks, TotalFuncs), then one
// counter for each instruction type.
#define HANDLE_INST(N, OPCODE, CLASS) +1
constexpr size_t kInstCountFeatureDimensionality = 3
#include "llvm/IR/Instruction.def"
    ;

using InstCountFeatureVector = std::array<int64_t, kInstCountFeatureDimensionality>;

class InstCount : public FunctionPass, public InstVisitor<InstCount> {
 public:
  // Return a vector of length kInstCountFeatureDimensionality with feature
  // values.
  static InstCountFeatureVector getFeatureVector(llvm::Module&);

  // Return a vector of length kInstCountFeatureDimensionality with the names
  // of each feature, in order.
  static std::array<std::string, kInstCountFeatureDimensionality> getFeatureNames();

 private:
  InstCount() : FunctionPass(ID) {}

  friend class InstVisitor<InstCount>;

  void visitFunction(Function& F) { ++TotalFuncs; }
  void visitBasicBlock(BasicBlock& BB) { ++TotalBlocks; }

#define HANDLE_INST(N, OPCODE, CLASS) \
  void visit##OPCODE(CLASS&) {        \
    ++Num##OPCODE##Inst;              \
    ++TotalInsts;                     \
  }

#include "llvm/IR/Instruction.def"

  void visitInstruction(Instruction& I) {
    errs() << "Instruction Count does not know about " << I;
    llvm_unreachable(nullptr);
  }

  // Declare a counter variable and a getter function.
#define COUNTER(name, unused_description) \
  int64_t name = 0;                       \
  int64_t get_##name() const { return name; }

  COUNTER(TotalInsts, "Number of instructions (of all types)");
  COUNTER(TotalBlocks, "Number of basic blocks");
  COUNTER(TotalFuncs, "Number of non-external functions");

  // Generate opcode counters.
#define HANDLE_INST(N, OPCODE, CLASS) COUNTER(Num##OPCODE##Inst, "Number of " #OPCODE " insts");

#include "llvm/IR/Instruction.def"

  static char ID;  // Pass identification, replacement for typeid

  bool runOnFunction(Function& F) override;
};

}  // namespace compiler_gym::llvm_service

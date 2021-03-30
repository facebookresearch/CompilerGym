// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/third_party/llvm/InstCount.h"

#include <glog/logging.h>

using namespace llvm;

namespace compiler_gym::llvm_service {

char InstCount::ID = 0;

bool InstCount::runOnFunction(Function& F) {
  visit(F);
  return false;
}

InstCountFeatureVector InstCount::getFeatureVector(llvm::Module& module) {
  InstCount pass;
  for (auto& function : module) {
    pass.runOnFunction(function);
  }

#define GET_COUNTER_VALUE(name) pass.get_##name()
#define HANDLE_INST(N, OPCODE, CLASS) GET_COUNTER_VALUE(Num##OPCODE##Inst),

  return {{
      GET_COUNTER_VALUE(TotalInsts),
      GET_COUNTER_VALUE(TotalBlocks),
      GET_COUNTER_VALUE(TotalFuncs),
#include "llvm/IR/Instruction.def"
  }};
}

std::array<std::string, kInstCountFeatureDimensionality> InstCount::getFeatureNames() {
#define HANDLE_INST(N, OPCODE, CLASS) #OPCODE,
  return {{
      "TotalInsts",
      "TotalBlocks",
      "TotalFuncs",
#include "llvm/IR/Instruction.def"
  }};
}

}  // namespace compiler_gym::llvm_service

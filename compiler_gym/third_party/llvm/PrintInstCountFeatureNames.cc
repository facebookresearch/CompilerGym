// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Print the name of each feature in the InstCount feature vector, in order.
#include <iostream>

#include "compiler_gym/third_party/llvm/InstCount.h"

using namespace compiler_gym::llvm_service;

int main(int argc, char** argv) {
  for (const auto& name : InstCount::getFeatureNames()) {
    std::cout << name << '\n';
  }
  return 0;
}

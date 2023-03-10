// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <glog/logging.h>

#include <fstream>
#include <iostream>

#include "lexed_ir.h"
#include "llvm_lexer.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  CHECK(argc == 2) << "Usage: unlex <token-id-path>";

  std::ifstream tokenFile(argv[1]);
  std::vector<llvm_lexer_token> tokens;

  if (tokenFile) {
    int id;
    std::string value;
    char comma;
    while (tokenFile >> id >> comma >> value) {
      llvm::lltok::Kind kind = static_cast<llvm::lltok::Kind>(id);
      llvm_lexer_token token = {kind, value};
      tokens.push_back(token);
    }
  }
  tokenFile.close();

  std::string ir = LexedIr::UnLex(tokens);
  std::cout << ir << std::endl;

  return 0;
}

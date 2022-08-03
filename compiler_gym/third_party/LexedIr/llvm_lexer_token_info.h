// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//=============================================================================
#include <string>
#include <vector>

#include "LLToken.h"

enum llvm_lexer_token_category { unknown, marker, keyword, uintval, strval, type, apfloat, apsint };

extern const std::vector<std::string> llvm_lexer_token_category_names;

struct llvm_lexer_token_info {
  llvm::lltok::Kind kind;
  llvm_lexer_token_category category;
  std::string name;
  std::string data;

  bool has_value() const {
    switch (category) {
      case unknown:
      case marker:
      case keyword:
        return false;
      case uintval:
      case strval:
      case type:
      case apfloat:
      case apsint:
        return true;
    };
  }
};

extern const std::vector<llvm_lexer_token_info> llvm_lexer_tokens;

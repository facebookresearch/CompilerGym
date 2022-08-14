// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//=============================================================================
#include <iostream>
#include <memory>

#include "LLLexer.h"
#include "escape.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/SystemUtils.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/IPO.h"
#include "llvm_lexer.h"

using namespace llvm;
using TokensVector = std::pair<std::pair<std::vector<int>, std::vector<std::string>>,
                               std::pair<std::vector<std::string>, std::vector<std::string>>>;

namespace LexedIr {

/// Create one token from the lexer
llvm_lexer_token LexOne(LLLexer& lexer) {
  lltok::Kind kind = lexer.getKind();
  std::string value;
  raw_string_ostream rso(value);

  switch (llvm_lexer_tokens[kind].category) {
    case keyword:
      break;
    case uintval: {
      rso << lexer.getUIntVal();
      break;
    }
    case strval: {
      value = escape(lexer.getStrVal(), true);
      break;
    }
    case type: {
      lexer.getTyVal()->print(rso);
      break;
    }
    case apfloat: {
      lexer.getAPFloatVal().print(rso);
      break;
    }
    case apsint: {
      lexer.getAPSIntVal().print(rso, false);
      break;
    }
    default: {
      break;
    }
  }
  return {kind, value};
}

/**
 * @brief Lex an LLVM-IR file (string format) into tokens using LLVM lexer.
 * @return A pair of vectors. First contains token indices (int), second contains string atoms.
 */
static TokensVector LexIR(const std::string& ir) {
  TokensVector tokens;

  LLVMContext ctx;
  SMDiagnostic err;
  SourceMgr sm;

  LLLexer lexer(ir, sm, err, ctx);

  std::unique_ptr<llvm_lexer_token> lexed;
  while (lexer.Lex() > lltok::Eof) {
    lexed = std::make_unique<llvm_lexer_token>(LexOne(lexer));
    tokens.first.first.push_back(lexed->kind);
    tokens.first.second.push_back(lexed->name());
    tokens.second.first.push_back(llvm_lexer_token_category_names[lexed->category()]);
    tokens.second.second.push_back(lexed->value);
  }
  return tokens;
}

/// Lex all the tokens in an IR
static std::string UnLex(const std::vector<llvm_lexer_token> tokens) {
  std::string ir;
  for (llvm_lexer_token token : tokens) ir += token.reconstruct() + " ";
  return ir;
}
};  // namespace LexedIr

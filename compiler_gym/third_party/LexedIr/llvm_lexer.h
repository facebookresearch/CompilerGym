// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

//=============================================================================
// This file is a small library to do llvm lexing
#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "LLToken.h"
#include "escape.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/IR/Type.h"
#include "llvm_lexer_token_info.h"

/// A simplified version of the LLVM assembly lexer
struct llvm_lexer_token {
  const llvm::lltok::Kind kind;
  const std::string value;

  const llvm_lexer_token_info& info() const { return llvm_lexer_tokens[kind]; }
  llvm_lexer_token_category category() const { return info().category; }
  std::string name() const { return info().name; }
  bool has_value() const { return info().has_value(); }

  /// Convert a token back into text
  std::string reconstruct() {
    switch (this->category()) {
      case marker:
      case keyword: {
        return info().data;
      }
      case uintval: {
        std::string s;
        if (kind != llvm::lltok::LabelID)
          s += info().data;
        s += value;
        if (kind == llvm::lltok::LabelID)
          s += info().data;
        return s;
      }
      case strval: {
        switch (kind) {
          case llvm::lltok::LabelStr: {
            return value + info().data;
          }
          case llvm::lltok::GlobalVar:
          case llvm::lltok::LocalVar:
          case llvm::lltok::EmissionKind:
          case llvm::lltok::NameTableKind:
          case llvm::lltok::StringConstant: {
            return info().data + '"' + escape(value, true) + '"';
          }
          default: {
            return info().data + value;
          }
        }
      }
      case type:
      case apfloat:
      case apsint:
        return value;
      default:
        return "<unknown>";
    }
  }
};

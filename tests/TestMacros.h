// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Testing macros.
#pragma once

#include <string>

// Expect that LLVM module passes verification.
// Usage: EXPECT_LLVM_MODULE_VALID(const_module_ref);
#define EXPECT_LLVM_MODULE_VALID(module)                             \
  do {                                                               \
    std::string errorMessage;                                        \
    llvm::raw_string_ostream rso(errorMessage);                      \
    if (llvm::verifyModule(module, &rso)) {                          \
      rso.flush();                                                   \
      FAIL() << "LLVM module failed verification: " << errorMessage; \
    }                                                                \
  } while (0);

// Expect that LLVM module fails verification.
// Usage: EXPECT_LLVM_MODULE_INVALID(const_module_ref);
#define EXPECT_LLVM_MODULE_INVALID(module)                    \
  do {                                                        \
    if (!llvm::verifyModule(module)) {                        \
      FAIL() << "Expected invalid LLVM module, but is valid"; \
    }                                                         \
  } while (0);

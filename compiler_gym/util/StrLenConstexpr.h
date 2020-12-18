// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <cstddef>

namespace compiler_gym::util {

// Calculate the length of a string literal at compile-time.
// E.g., strLen("abc") -> 3.
template <typename T>
size_t constexpr strLen(const T* str) {
  return *str ? 1 + strLen(str + 1) : 0;
}

}  // namespace compiler_gym::util

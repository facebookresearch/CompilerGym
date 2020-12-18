// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include "compiler_gym/util/StrLenConstexpr.h"

namespace compiler_gym::util {
namespace {

TEST(StrLenConstexpr, emptyString) { EXPECT_EQ(strLen(""), 0); }

TEST(StrLenConstexpr, strings) {
  EXPECT_EQ(strLen(" "), 1);
  EXPECT_EQ(strLen("a"), 1);
  EXPECT_EQ(strLen("\t"), 1);
  EXPECT_EQ(strLen("\n"), 1);
  EXPECT_EQ(strLen("abc"), 3);
}

}  // anonymous namespace
}  // namespace compiler_gym::util

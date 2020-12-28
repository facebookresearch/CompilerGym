// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include <magic_enum.hpp>

#include "compiler_gym/envs/llvm/service/RewardSpaces.h"
#include "tests/TestMacros.h"

using namespace ::testing;

namespace compiler_gym::llvm_service {
namespace {

TEST(RewardSpacesTest, getLlvmRewardSpaceList) {
  const auto spaces = getLlvmRewardSpaceList();

  ASSERT_EQ(spaces.size(), 6);

  EXPECT_EQ(spaces[0].name(), "IrInstructionCount");
  EXPECT_FALSE(spaces[0].range().has_min());
  EXPECT_EQ(spaces[0].range().max().value(), 0);

  EXPECT_EQ(spaces[1].name(), "IrInstructionCountO3");
  EXPECT_EQ(spaces[1].range().min().value(), 0);
  EXPECT_FALSE(spaces[1].range().has_max());

  EXPECT_EQ(spaces[2].name(), "IrInstructionCountOz");
  EXPECT_EQ(spaces[2].range().min().value(), 0);
  EXPECT_FALSE(spaces[2].range().has_max());

  EXPECT_EQ(spaces[3].name(), "ObjectTextSizeBytes");
  EXPECT_FALSE(spaces[3].range().has_min());
  EXPECT_EQ(spaces[3].range().max().value(), 0);

  EXPECT_EQ(spaces[4].name(), "ObjectTextSizeO3");
  EXPECT_FALSE(spaces[4].range().has_min());
  EXPECT_EQ(spaces[4].range().max().value(), 0);

  EXPECT_EQ(spaces[5].name(), "ObjectTextSizeOz");
  EXPECT_FALSE(spaces[5].range().has_min());
  EXPECT_EQ(spaces[5].range().max().value(), 0);
}

}  // anonymous namespace
}  // namespace compiler_gym::llvm_service

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

  auto space = spaces.begin();
  EXPECT_EQ(space->name(), "IrInstructionCount");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_FALSE(space->range().has_max());

  ++space;
  EXPECT_EQ(space->name(), "IrInstructionCountNorm");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_EQ(space->range().max().value(), 1);

  ++space;
  EXPECT_EQ(space->name(), "IrInstructionCountO3");
  EXPECT_EQ(space->range().min().value(), 0);
  EXPECT_FALSE(space->range().has_max());

  ++space;
  EXPECT_EQ(space->name(), "IrInstructionCountOz");
  EXPECT_EQ(space->range().min().value(), 0);
  EXPECT_FALSE(space->range().has_max());

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeBytes");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_FALSE(space->range().has_max());

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeNorm");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_EQ(space->range().max().value(), 1);

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeO3");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_FALSE(space->range().has_max());

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeOz");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_FALSE(space->range().has_max());

  ++space;
  EXPECT_EQ(space, spaces.end());
}

}  // anonymous namespace
}  // namespace compiler_gym::llvm_service

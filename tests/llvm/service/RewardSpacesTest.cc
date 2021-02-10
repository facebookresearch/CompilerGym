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
  EXPECT_FALSE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_FALSE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space->name(), "IrInstructionCountNorm");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_EQ(space->range().max().value(), 1);
  EXPECT_FALSE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_FALSE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space->name(), "IrInstructionCountO3");
  EXPECT_EQ(space->range().min().value(), 0);
  EXPECT_FALSE(space->range().has_max());
  EXPECT_TRUE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_FALSE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space->name(), "IrInstructionCountOz");
  EXPECT_EQ(space->range().min().value(), 0);
  EXPECT_FALSE(space->range().has_max());
  EXPECT_TRUE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_FALSE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeBytes");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_FALSE(space->range().has_max());
  EXPECT_FALSE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_TRUE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeNorm");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_EQ(space->range().max().value(), 1);
  EXPECT_FALSE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_TRUE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeO3");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_FALSE(space->range().has_max());
  EXPECT_TRUE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_TRUE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space->name(), "ObjectTextSizeOz");
  EXPECT_FALSE(space->range().has_min());
  EXPECT_FALSE(space->range().has_max());
  EXPECT_TRUE(space->has_success_threshold());
  EXPECT_TRUE(space->deterministic());
  EXPECT_TRUE(space->platform_dependent());
  EXPECT_EQ(space->default_value(), 0);
  EXPECT_TRUE(space->default_negates_returns());

  ++space;
  EXPECT_EQ(space, spaces.end());
}

}  // anonymous namespace
}  // namespace compiler_gym::llvm_service

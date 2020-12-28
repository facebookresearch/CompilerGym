// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include <magic_enum.hpp>

#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"

using namespace ::testing;

namespace compiler_gym::llvm_service {
namespace {

TEST(ObservationSpacesTest, getLlvmObservationSpaceList) {
  const auto spaces = getLlvmObservationSpaceList();

  ASSERT_EQ(spaces.size(), 13);

  EXPECT_EQ(spaces[0].name(), "Ir");
  EXPECT_EQ(spaces[0].string_size_range().min().value(), 0);
  EXPECT_FALSE(spaces[0].string_size_range().has_max());

  EXPECT_EQ(spaces[1].name(), "BitcodeFile");
  EXPECT_EQ(spaces[1].string_size_range().min().value(), 0);
  EXPECT_EQ(spaces[1].string_size_range().max().value(), 4096);

  EXPECT_EQ(spaces[2].name(), "Autophase");
  ASSERT_EQ(spaces[2].int64_range_list().range_size(), 56);
  for (int i = 0; i < /* autophase feature vector dim: */ 56; ++i) {
    EXPECT_EQ(spaces[2].int64_range_list().range(i).min().value(), 0);
    EXPECT_FALSE(spaces[2].int64_range_list().range(i).has_max());
  }

  EXPECT_EQ(spaces[3].name(), "Programl");

  EXPECT_EQ(spaces[4].name(), "CpuInfo");

  EXPECT_EQ(spaces[5].name(), "IrInstructionCount");
  EXPECT_EQ(spaces[6].name(), "IrInstructionCountO0");
  EXPECT_EQ(spaces[7].name(), "IrInstructionCountO3");
  EXPECT_EQ(spaces[8].name(), "IrInstructionCountOz");

  EXPECT_EQ(spaces[9].name(), "ObjectTextSizeBytes");
  EXPECT_EQ(spaces[10].name(), "ObjectTextSizeO0");
  EXPECT_EQ(spaces[11].name(), "ObjectTextSizeO3");
  EXPECT_EQ(spaces[12].name(), "ObjectTextSizeOz");
}

}  // anonymous namespace
}  // namespace compiler_gym::llvm_service

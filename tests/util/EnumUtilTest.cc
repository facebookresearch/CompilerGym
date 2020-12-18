// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include <optional>
#include <stdexcept>

#include "compiler_gym/util/EnumUtil.h"
#include "tests/TestMacros.h"

namespace compiler_gym::util {
namespace {

enum class Fruit {  // Deliberately grotesque code style.
  APPLES,
  UNRIPE_bAnAnAs,
  is_tomato_even_a_fruit,
};

TEST(EnumUtilTest, enumNameToPascalCase) {
  EXPECT_EQ(enumNameToPascalCase(Fruit::APPLES), "Apples");
  EXPECT_EQ(enumNameToPascalCase(Fruit::UNRIPE_bAnAnAs), "UnripeBananas");
  EXPECT_EQ(enumNameToPascalCase(Fruit::is_tomato_even_a_fruit), "IsTomatoEvenAFruit");
}

TEST(EnumUtilTest, optionalEnumNameToPascalCase) {
  EXPECT_EQ(enumNameToPascalCase<Fruit>(std::nullopt), "None");
}

TEST(EnumUtilTest, optionalEnumValues) {
  const auto values = optionalEnumValues<Fruit>();
  ASSERT_EQ(values.size(), 4);
  EXPECT_EQ(values[0], std::nullopt);
  EXPECT_EQ(values[1], Fruit::APPLES);
  EXPECT_EQ(values[2], Fruit::UNRIPE_bAnAnAs);
  EXPECT_EQ(values[3], Fruit::is_tomato_even_a_fruit);
}

TEST(EnumUtilTest, intToEnumNegativeValue) {
  Fruit fruit;
  auto status = intToEnum(-1, &fruit);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_message(), "Fruit(-1) is not in range [0, 3)");
}

TEST(EnumUtilTest, intToEnumOutOfRangeValue) {
  Fruit fruit;
  auto status = intToEnum(100, &fruit);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_message(), "Fruit(100) is not in range [0, 3)");
}

TEST(EnumUtilTest, intToEnum) {
  Fruit fruit;
  auto status = intToEnum(0, &fruit);
  EXPECT_TRUE(status.ok());
  EXPECT_EQ(fruit, Fruit::APPLES);
}

}  // anonymous namespace
}  // namespace compiler_gym::util

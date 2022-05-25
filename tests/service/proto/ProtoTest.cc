// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include <utility>

#include "compiler_gym/service/proto/Proto.h"

namespace compiler_gym {

struct SpaceContainsEventCheckerTest : ::testing::Test {
  void SetUp() override { checker = makeDefaultSpaceContainsEventChecker(); }

  SpaceContainsEventChecker checker;
};

TEST_F(SpaceContainsEventCheckerTest, SpaceListContainsTrue) {
  Space subspace;
  subspace.mutable_int64_value();
  Space space;
  space.mutable_space_list()->mutable_space()->Add(std::move(subspace));
  Event subevent;
  subevent.set_int64_value(0);
  Event event;
  event.mutable_event_list()->mutable_event()->Add(std::move(subevent));
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, SpaceListContainsFalse) {
  Space subspace;
  subspace.mutable_int64_value();
  Space space;
  space.mutable_space_list()->mutable_space()->Add(std::move(subspace));
  Event event;
  event.mutable_event_list();
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, SpaceDictContainsTrue) {
  Space subspace;
  subspace.mutable_int64_value();
  Space space;
  (*space.mutable_space_dict()->mutable_space())["subspace"] = std::move(subspace);
  Event subevent;
  subevent.set_int64_value(0);
  Event event;
  (*event.mutable_event_dict()->mutable_event())["subspace"] = std::move(subevent);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, SpaceDictContainsFalse) {
  Space subspace;
  Space space;
  (*space.mutable_space_dict()->mutable_space())["subspace1"] = std::move(subspace);
  Event subevent;
  Event event;
  (*event.mutable_event_dict()->mutable_event())["subspace2"] = std::move(subevent);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DiscreteContainsTrue) {
  Space space;
  space.mutable_discrete()->set_n(3);
  Event event;
  event.set_int64_value(2);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DiscreteContainsFalse) {
  Space space;
  space.mutable_discrete()->set_n(3);
  Event event;
  event.set_int64_value(3);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, NamedDiscreteContainsTrue) {
  Space space;
  space.mutable_named_discrete()->mutable_name()->Add(std::string("name1"));
  Event event;
  event.set_int64_value(0);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, NamedDiscreteContainsFalse) {
  Space space;
  space.mutable_named_discrete()->mutable_name()->Add(std::string("name1"));
  Event event;
  event.set_int64_value(1);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, BooleanContainsTrue) {
  Space space;
  space.mutable_boolean_value();
  Event event;
  event.set_boolean_value(true);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, BooleanContainsFalse) {
  Space space;
  space.mutable_boolean_value()->set_max(false);
  Event event;
  event.set_boolean_value(true);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, Int64ContainsTrue) {
  Space space;
  space.mutable_int64_value()->set_min(0);
  space.mutable_int64_value()->set_max(0);
  Event event;
  event.set_int64_value(0);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, Int64ContainsFalse) {
  Space space;
  space.mutable_int64_value()->set_min(1);
  Event event;
  event.set_int64_value(0);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, FloatContainsTrue) {
  Space space;
  space.mutable_float_value()->set_min(0);
  space.mutable_float_value()->set_max(0);
  Event event;
  event.set_float_value(0);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, FloatContainsFalse) {
  Space space;
  space.mutable_float_value();
  Event event;
  event.set_int64_value(0);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DoubleContainsTrue) {
  Space space;
  space.mutable_double_value();
  Event event;
  event.set_double_value(0);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DoubleContainsFalse) {
  Space space;
  space.mutable_double_value()->set_max(-1);
  Event event;
  event.set_double_value(0);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, StringContainsTrue) {
  Space space;
  space.mutable_string_value()->mutable_length_range()->set_max(3);
  Event event;
  event.set_string_value("val");
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, StringContainsFalse) {
  Space space;
  space.mutable_string_value()->mutable_length_range()->set_max(3);
  Event event;
  event.set_string_value("long val");
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, BooleanSequenceContainsTrue) {
  Space space;
  space.mutable_boolean_sequence()->mutable_scalar_range()->set_max(false);
  Event event;
  *event.mutable_boolean_tensor()->mutable_shape()->Add() = 1;
  *event.mutable_boolean_tensor()->mutable_value()->Add() = false;
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, BooleanSequenceContainsFalse) {
  Space space;
  space.mutable_boolean_sequence()->mutable_scalar_range()->set_max(false);
  Event event;
  *event.mutable_boolean_tensor()->mutable_shape()->Add() = 1;
  *event.mutable_boolean_tensor()->mutable_value()->Add() = true;
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, ByteSequenceContainsTrue) {
  Space space;
  space.mutable_byte_sequence()->mutable_scalar_range()->set_max(2);
  space.mutable_byte_sequence()->mutable_length_range()->set_max(1);
  Event event;
  *event.mutable_byte_tensor()->mutable_shape()->Add() = 1;
  *event.mutable_byte_tensor()->mutable_value() += 2;
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, ByteSequenceContainsFalse) {
  Space space;
  space.mutable_byte_sequence()->mutable_scalar_range()->set_max(-1);
  space.mutable_byte_sequence()->mutable_scalar_range()->set_max(2);
  space.mutable_byte_sequence()->mutable_length_range()->set_min(1);
  space.mutable_byte_sequence()->mutable_length_range()->set_max(2);
  Event event;
  *event.mutable_byte_tensor()->mutable_shape()->Add() = 1;
  *event.mutable_byte_tensor()->mutable_value() += 3;
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, Int64SequenceContainsTrue) {
  Space space;
  space.mutable_int64_sequence()->mutable_scalar_range()->set_max(2);
  space.mutable_int64_sequence()->mutable_length_range()->set_max(1);
  Event event;
  *event.mutable_int64_tensor()->mutable_shape()->Add() = 1;
  event.mutable_int64_tensor()->mutable_value()->Add(2);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, Int64SequenceContainsFalse) {
  Space space;
  space.mutable_int64_sequence()->mutable_length_range()->set_max(1);
  Event event;
  *event.mutable_int64_tensor()->mutable_shape()->Add() = 2;
  event.mutable_int64_tensor()->mutable_value()->Add(0);
  event.mutable_int64_tensor()->mutable_value()->Add(0);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, FloatSequenceContainsTrue) {
  Space space;
  space.mutable_float_sequence()->mutable_length_range()->set_max(2);
  Event event;
  *event.mutable_float_tensor()->mutable_shape()->Add() = 1;
  event.mutable_float_tensor()->mutable_value()->Add(2);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, FloatSequenceContainsFalse) {
  Space space;
  space.mutable_float_sequence()->mutable_length_range()->set_min(1);
  Event event;
  *event.mutable_float_tensor()->mutable_shape()->Add() = 0;
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DoubleSequenceContainsTrue) {
  Space space;
  space.mutable_double_sequence()->mutable_scalar_range()->set_min(2);
  space.mutable_double_sequence()->mutable_scalar_range()->set_max(2);
  Event event;
  *event.mutable_double_tensor()->mutable_shape()->Add() = 1;
  event.mutable_double_tensor()->mutable_value()->Add(2);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DoubleSequenceContainsFalse) {
  Space space;
  space.mutable_double_sequence()->mutable_scalar_range()->set_min(1);
  Event event;
  *event.mutable_double_tensor()->mutable_shape()->Add() = 1;
  event.mutable_double_tensor()->mutable_value()->Add(-1);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, StringSequenceContainsTrue) {
  Space space;
  space.mutable_string_sequence()->mutable_length_range()->set_min(1);
  space.mutable_string_sequence()->mutable_length_range()->set_max(1);
  Event event;
  *event.mutable_string_tensor()->mutable_shape()->Add() = 1;
  event.mutable_string_tensor()->mutable_value()->Add("str");
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, StringSequenceContainsFalse) {
  Space space;
  space.mutable_string_sequence()->mutable_length_range()->set_min(1);
  space.mutable_string_sequence()->mutable_length_range()->set_max(1);
  Event event;
  *event.mutable_string_tensor()->mutable_shape()->Add() = 2;
  event.mutable_string_tensor()->mutable_value()->Add("str1");
  event.mutable_string_tensor()->mutable_value()->Add("str2");
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, SpaceSequenceContainsTrue) {
  Space space;
  space.mutable_space_sequence()->mutable_length_range()->set_max(1);
  space.mutable_space_sequence()->mutable_space()->mutable_int64_value();
  Event event;
  Event subevent;
  subevent.set_int64_value(0);
  event.mutable_event_list()->mutable_event()->Add(std::move(subevent));
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, SpaceSequenceContainsFalse) {
  Space space;
  space.mutable_space_sequence()->mutable_length_range()->set_max(1);
  space.mutable_space_sequence()->mutable_space()->mutable_int64_value();
  Event event;
  Event subevent;
  subevent.set_float_value(0);
  event.mutable_event_list()->mutable_event()->Add(std::move(subevent));
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, BooleanBoxContainsTrue) {
  Space space;
  space.mutable_boolean_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_boolean_box()->mutable_low()->mutable_value()->Add(false);
  space.mutable_boolean_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_boolean_box()->mutable_high()->mutable_value()->Add(false);
  Event event;
  event.mutable_boolean_tensor()->mutable_shape()->Add(1);
  event.mutable_boolean_tensor()->mutable_value()->Add(false);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, BooleanBoxContainsFalse) {
  Space space;
  space.mutable_boolean_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_boolean_box()->mutable_low()->mutable_value()->Add(false);
  space.mutable_boolean_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_boolean_box()->mutable_high()->mutable_value()->Add(false);
  Event event;
  event.mutable_boolean_tensor()->mutable_shape()->Add(1);
  event.mutable_boolean_tensor()->mutable_value()->Add(true);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, ByteBoxContainsTrue) {
  Space space;
  space.mutable_byte_box()->mutable_low()->mutable_shape()->Add(1);
  *space.mutable_byte_box()->mutable_low()->mutable_value() += 1;
  space.mutable_byte_box()->mutable_high()->mutable_shape()->Add(1);
  *space.mutable_byte_box()->mutable_high()->mutable_value() += 1;
  Event event;
  event.mutable_byte_tensor()->mutable_shape()->Add(1);
  *event.mutable_byte_tensor()->mutable_value() += 1;
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, ByteBoxContainsFalse) {
  Space space;
  space.mutable_byte_box()->mutable_low()->mutable_shape()->Add(1);
  *space.mutable_byte_box()->mutable_low()->mutable_value() += 1;
  space.mutable_byte_box()->mutable_high()->mutable_shape()->Add(1);
  *space.mutable_byte_box()->mutable_high()->mutable_value() += 1;
  Event event;
  event.mutable_byte_tensor()->mutable_shape()->Add(1);
  *event.mutable_byte_tensor()->mutable_value() += 2;
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, Int64BoxContainsTrue) {
  Space space;
  space.mutable_int64_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_int64_box()->mutable_low()->mutable_value()->Add(2);
  space.mutable_int64_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_int64_box()->mutable_high()->mutable_value()->Add(2);
  Event event;
  event.mutable_int64_tensor()->mutable_shape()->Add(1);
  event.mutable_int64_tensor()->mutable_value()->Add(2);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, Int64BoxContainsFalse) {
  Space space;
  space.mutable_int64_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_int64_box()->mutable_low()->mutable_value()->Add(2);
  space.mutable_int64_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_int64_box()->mutable_high()->mutable_value()->Add(2);
  Event event;
  event.mutable_int64_tensor()->mutable_shape()->Add(2);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, FloatBoxContainsTrue) {
  Space space;
  space.mutable_float_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_float_box()->mutable_low()->mutable_value()->Add(2);
  space.mutable_float_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_float_box()->mutable_high()->mutable_value()->Add(2);
  Event event;
  event.mutable_float_tensor()->mutable_shape()->Add(1);
  event.mutable_float_tensor()->mutable_value()->Add(2);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, FloatBoxContainsFalse) {
  Space space;
  space.mutable_float_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_float_box()->mutable_low()->mutable_value()->Add(2);
  space.mutable_float_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_float_box()->mutable_high()->mutable_value()->Add(2);
  Event event;
  event.mutable_float_tensor()->mutable_shape()->Add(1);
  event.mutable_float_tensor()->mutable_value()->Add(3);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DoubleBoxContainsTrue) {
  Space space;
  space.mutable_double_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_double_box()->mutable_low()->mutable_value()->Add(2);
  space.mutable_double_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_double_box()->mutable_high()->mutable_value()->Add(2);
  Event event;
  event.mutable_double_tensor()->mutable_shape()->Add(1);
  event.mutable_double_tensor()->mutable_value()->Add(2);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, DoubleBoxContainsFalse) {
  Space space;
  space.mutable_double_box()->mutable_low()->mutable_shape()->Add(1);
  space.mutable_double_box()->mutable_low()->mutable_value()->Add(2);
  space.mutable_double_box()->mutable_high()->mutable_shape()->Add(1);
  space.mutable_double_box()->mutable_high()->mutable_value()->Add(2);
  Event event;
  event.mutable_double_tensor()->mutable_shape()->Add(1);
  event.mutable_double_tensor()->mutable_value()->Add(3);
  EXPECT_FALSE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, SpaceUnionContainsTrue) {
  Space subspace;
  subspace.mutable_int64_value();
  Space space;
  *space.mutable_space_union()->mutable_space()->Add() = std::move(subspace);
  Event event;
  event.set_int64_value(1);
  EXPECT_TRUE(checker.contains(space, event));
}

TEST_F(SpaceContainsEventCheckerTest, SpaceUnionContainsFalse) {
  Space subspace;
  subspace.mutable_int64_value();
  Space space;
  *space.mutable_space_union()->mutable_space()->Add() = std::move(subspace);
  Event event;
  event.set_double_value(1);
  EXPECT_FALSE(checker.contains(space, event));
}

}  // namespace compiler_gym

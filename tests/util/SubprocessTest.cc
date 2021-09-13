// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include "compiler_gym/util/Subprocess.h"

namespace compiler_gym::util {
namespace {

using grpc::StatusCode;

TEST(Subprocess, checkCallTrue) { EXPECT_TRUE(checkCall("true", 60, ".").ok()); }

TEST(Subprocess, checkCallFalse) {
  const auto status = checkCall("false", 60, ".");
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "Command 'false' failed with exit code: 1");
}

TEST(Subprocess, checkCallShellPipe) { EXPECT_TRUE(checkCall("echo Hello | cat", 60, ".").ok()); }

TEST(Subprocess, checkCallConcatenatedCommands) {
  const auto status = checkCall("echo Hello ; echo Foo", 60, ".");
  EXPECT_TRUE(status.ok());
}

TEST(Subprocess, checkCallFirstElementInPipeFails) {
  const auto status = checkCall("true | false", 60, ".");
  EXPECT_TRUE(status.ok());
}

TEST(Subprocess, checkCallLastElementInPipeFails) {
  const auto status = checkCall("false | true", 60, ".");
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "Command 'false | true' failed with exit code: 1");
}

TEST(Subprocess, checkCallTimeout) {
  const auto status = checkCall("sleep 10", 1, ".");
  EXPECT_EQ(status.error_code(), StatusCode::DEADLINE_EXCEEDED);
  EXPECT_EQ(status.error_message(), "Command 'sleep 10' failed to complete within 1 seconds");
}

TEST(Subprocess, checkCallWorkingDirNotFound) {
  const auto status = checkCall("true", 60, "not/a/real/path");
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "Failed to set working directory: not/a/real/path");
}

}  // anonymous namespace
}  // namespace compiler_gym::util

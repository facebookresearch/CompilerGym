// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include "compiler_gym/util/Subprocess.h"

namespace compiler_gym::util {
namespace {

using grpc::StatusCode;

TEST(Subprocess, true_) { EXPECT_TRUE(checkCall("true", 60, ".").ok()); }

TEST(Subprocess, false_) {
  const auto status = checkCall("false", 60, ".");
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "Command 'false' failed with exit code: 1. Stderr:\n");
}

TEST(Subprocess, shellPipe) { EXPECT_TRUE(checkCall("echo Hello | cat", 60, ".").ok()); }

TEST(Subprocess, lastElementInPipeFails) {
  const auto status = checkCall("echo Hello >&2 | false", 60, ".");
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(),
            "Command 'echo Hello >&2 | false' failed with exit code: 1. Stderr:\nHello\n");
}

TEST(Subprocess, firstElementInPipeFails) {
  const auto status = checkCall("false | echo Hello >&2", 60, ".");
  EXPECT_TRUE(status.ok());
}

TEST(Subprocess, timeout) {
  const auto status = checkCall("sleep 10", 1, ".");
  EXPECT_EQ(status.error_code(), StatusCode::DEADLINE_EXCEEDED);
  EXPECT_EQ(status.error_message(), "Command 'sleep 10' failed to complete within 1 seconds");
}

TEST(Subprocess, workingDirNotFound) {
  const auto status = checkCall("true", 60, "not/a/real/path");
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(),
            "Command 'true' failed with error: chdir failed : No such file or directory");
}

}  // anonymous namespace
}  // namespace compiler_gym::util

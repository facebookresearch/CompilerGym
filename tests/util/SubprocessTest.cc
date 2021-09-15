// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include "compiler_gym/util/Subprocess.h"

namespace compiler_gym::util {
namespace {

using grpc::StatusCode;

TEST(Subprocess, checkCallTrue) {
  Command proto;
  proto.add_argument("true");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  EXPECT_TRUE(cmd.checkCall().ok());
}

TEST(Subprocess, checkCallFalse) {
  Command proto;
  proto.add_argument("false");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  const auto status = cmd.checkCall();
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "Command 'false' failed with exit code 1");
}

TEST(Subprocess, checkCallTwoArguments) {
  Command proto;
  proto.add_argument("echo");
  proto.add_argument("Hello");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  EXPECT_TRUE(cmd.checkCall().ok());
}

TEST(Subprocess, checkCallShellPipe) {
  Command proto;
  proto.add_argument("echo");
  proto.add_argument("Hello");
  proto.add_argument("|");
  proto.add_argument("cat");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  EXPECT_TRUE(cmd.checkCall().ok());
}

TEST(Subprocess, checkCallConcatenatedCommands) {
  Command proto;
  proto.add_argument("echo");
  proto.add_argument("Hello;");
  proto.add_argument("echo");
  proto.add_argument("Foo");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  EXPECT_TRUE(cmd.checkCall().ok());
}

TEST(Subprocess, checkCallFirstElementInPipeFails) {
  // NOTE(cummins): This demonstrates a deficiency in the API, as ideally
  // pipefail option should be set and this command should fail.
  Command proto;
  proto.add_argument("false");
  proto.add_argument("|");
  proto.add_argument("true");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  EXPECT_TRUE(cmd.checkCall().ok());
}

TEST(Subprocess, checkCallLastElementInPipeFails) {
  Command proto;
  proto.add_argument("true");
  proto.add_argument("|");
  proto.add_argument("false");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  const auto status = cmd.checkCall();
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(), "Command 'true | false' failed with exit code 1");
}

TEST(Subprocess, failingCommandStderrCapture) {
  Command proto;
  proto.add_argument("echo");
  proto.add_argument("Hello");
  proto.add_argument(">&2");
  proto.add_argument("|");
  proto.add_argument("false");
  proto.set_timeout_seconds(10);
  LocalShellCommand cmd(proto);

  const auto status = cmd.checkCall();
  EXPECT_EQ(status.error_code(), StatusCode::INTERNAL);
  EXPECT_EQ(status.error_message(),
            "Command 'echo Hello >&2 | false' failed with exit code 1: Hello\n");
}

TEST(Subprocess, checkCallTimeout) {
  Command proto;
  proto.add_argument("sleep");
  proto.add_argument("10");
  proto.set_timeout_seconds(1);
  LocalShellCommand cmd(proto);

  const auto status = cmd.checkCall();
  EXPECT_EQ(status.error_code(), StatusCode::DEADLINE_EXCEEDED);
  EXPECT_EQ(status.error_message(), "Command 'sleep 10' failed to complete within 1 seconds");
}

TEST(Subprocess, checkOutputEnvironmentVariable) {
  Command proto;
  proto.add_argument("echo");
  proto.add_argument("Hello $NAME");
  (*proto.mutable_env())["NAME"] = "Chris";
  proto.set_timeout_seconds(1);
  LocalShellCommand cmd(proto);

  std::string stdout;
  EXPECT_TRUE(cmd.checkOutput(stdout).ok());
  EXPECT_EQ(stdout, "Hello Chris\n");
}

}  // anonymous namespace
}  // namespace compiler_gym::util

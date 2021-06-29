// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <boost/filesystem.hpp>

namespace compiler_gym::util {

/**
 * Run the given command in a subshell and discard its output.
 *
 * @param cmd The command to execute as a string.
 * @param timeoutSeconds The number of seconds to wait for the command to
 *     terminate before failing with an error.
 * @return `OK` on success, `DEADLINE_EXCEEDED` on timeout, or `INTERNAL` if the
 *    command returns with a non-zero returncode.
 */
grpc::Status checkCall(const std::string& cmd, int timeoutSeconds,
                       const boost::filesystem::path& workingDir);

}  // namespace compiler_gym::util

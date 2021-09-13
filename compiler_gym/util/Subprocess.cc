// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/util/Subprocess.h"

#include <fmt/format.h>
#include <unistd.h>

#include <boost/process.hpp>
#include <future>

namespace compiler_gym::util {

using grpc::Status;
using grpc::StatusCode;
namespace fs = boost::filesystem;
namespace bp = boost::process;

Status checkCall(const std::string& cmd, int timeoutSeconds, const fs::path& workingDir) {
  if (chdir(workingDir.string().c_str())) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to set working directory: {}", workingDir.string()));
  }

  try {
    bp::child process(cmd, bp::std_out > bp::null, bp::std_err > bp::null);

    if (!process.wait_for(std::chrono::seconds(timeoutSeconds))) {
      return Status(
          StatusCode::DEADLINE_EXCEEDED,
          fmt::format("Command '{}' failed to complete within {} seconds", cmd, timeoutSeconds));
    }

    if (process.exit_code()) {
      return Status(StatusCode::INTERNAL, fmt::format("Command '{}' failed with exit code: {}", cmd,
                                                      process.exit_code()));
    }
  } catch (bp::process_error& e) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Command '{}' failed with error: {}", cmd, e.what()));
  }

  return Status::OK;
}

}  // namespace compiler_gym::util

// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/util/Subprocess.h"

#include <fmt/format.h>

#include <chrono>
#include <condition_variable>
#include <subprocess/subprocess.hpp>
#include <thread>

namespace compiler_gym::util {

using grpc::Status;
using grpc::StatusCode;
namespace fs = boost::filesystem;

Status checkCall(const std::string& cmd, int timeoutSeconds, const fs::path& workingDir) {
  std::mutex m;
  std::condition_variable cv;
  Status returnStatus(StatusCode::INTERNAL, "Unknown error");

  std::thread t([&cmd, &workingDir, &cv, &returnStatus]() {
    try {
      auto run = subprocess::Popen(cmd, subprocess::output{subprocess::PIPE},
                                   subprocess::error{subprocess::PIPE}, subprocess::shell{true},
                                   subprocess::cwd{workingDir.string()});
      const auto runOutput = run.communicate();
      if (run.retcode()) {
        const std::string error(runOutput.second.buf.begin(), runOutput.second.buf.end());
        returnStatus = Status(StatusCode::INTERNAL,
                              fmt::format("Command '{}' failed with exit code: {}. Stderr:\n{}",
                                          cmd, run.retcode(), error));
      } else {
        returnStatus = Status::OK;
      }
    } catch (subprocess::CalledProcessError& e) {
      returnStatus = Status(StatusCode::INTERNAL,
                            fmt::format("Command '{}' failed with error: {}", cmd, e.what()));
    }
    cv.notify_one();
  });
  t.detach();

  {
    std::unique_lock<std::mutex> lock(m);
    if (cv.wait_for(lock, std::chrono::seconds(timeoutSeconds)) == std::cv_status::timeout) {
      return Status(
          StatusCode::DEADLINE_EXCEEDED,
          fmt::format("Command '{}' failed to complete within {} seconds", cmd, timeoutSeconds));
    }
  }

  return returnStatus;
}

}  // namespace compiler_gym::util

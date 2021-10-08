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

namespace {

/**
 * Merge the given map<string, string> environment variables into a copy of the
 * current process environment.
 */
bp::environment makeEnvironment(const google::protobuf::Map<std::string, std::string>& userEnv) {
  bp::environment env{boost::this_process::environment()};
  for (const auto& item : userEnv) {
    env[item.first] = item.second;
  }
  return env;
}

/**
 * Convert a list of strings to boost::filesystem::path objects.
 */
std::vector<fs::path> makePaths(const google::protobuf::RepeatedPtrField<std::string>& strings) {
  std::vector<fs::path> paths;
  for (const auto& path : strings) {
    paths.push_back(fs::path(path));
  }
  return paths;
}

}  // anonymous namespace

LocalShellCommand::LocalShellCommand(const Command& cmd)
    : proto_(cmd),
      arguments_({cmd.argument().begin(), cmd.argument().end()}),
      timeout_(std::chrono::seconds(cmd.timeout_seconds())),
      env_(makeEnvironment(cmd.env())),
      infiles_(makePaths(cmd.infile())),
      outfiles_(makePaths(cmd.outfile())) {}

Status LocalShellCommand::checkCall() const {
  boost::asio::io_context stderrStream;
  std::future<std::string> stderrFuture;

  try {
    bp::child process(arguments(), bp::std_out > bp::null, bp::std_err > stderrFuture, bp::shell,
                      stderrStream, env());
    if (!wait_for(process, timeout())) {
      return Status(StatusCode::DEADLINE_EXCEEDED,
                    fmt::format("Command '{}' failed to complete within {} seconds", commandline(),
                                timeoutSeconds()));
    }
    stderrStream.run();

    if (process.exit_code()) {
      const std::string stderr = stderrFuture.get();
      if (stderr.size()) {
        return Status(StatusCode::INTERNAL,
                      fmt::format("Command '{}' failed with exit code {}: {}", commandline(),
                                  process.exit_code(), stderr));
      } else {
        return Status(StatusCode::INTERNAL, fmt::format("Command '{}' failed with exit code {}",
                                                        commandline(), process.exit_code()));
      }
    }
  } catch (bp::process_error& e) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Command '{}' failed with error: {}", commandline(), e.what()));
  }

  return Status::OK;
}

Status LocalShellCommand::checkOutput(std::string& stdout) const {
  try {
    boost::asio::io_context stdoutStream;
    std::future<std::string> stdoutFuture;

    bp::child process(arguments(), bp::std_in.close(), bp::std_out > stdoutFuture,
                      bp::std_err > bp::null, bp::shell, stdoutStream, env());

    if (!wait_for(process, timeout())) {
      return Status(StatusCode::DEADLINE_EXCEEDED,
                    fmt::format("Command '{}' failed to complete within {} seconds", commandline(),
                                timeoutSeconds()));
    }
    stdoutStream.run();

    if (process.exit_code()) {
      return Status(StatusCode::INTERNAL, fmt::format("Command '{}' failed with exit code {}",
                                                      commandline(), process.exit_code()));
    }

    stdout = stdoutFuture.get();
  } catch (bp::process_error& e) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Command '{}' failed with error: {}", commandline(), e.what()));
  }

  return Status::OK;
}

Status LocalShellCommand::checkInfiles() const {
  for (const auto& infile : infiles()) {
    if (!fs::exists(infile)) {
      return Status(StatusCode::INTERNAL, fmt::format("Command '{}' missing required file: {}",
                                                      commandline(), infile.string()));
    }
  }
  return Status::OK;
}

Status LocalShellCommand::checkOutfiles() const {
  for (const auto& outfile : outfiles()) {
    if (!fs::exists(outfile)) {
      return Status(StatusCode::INTERNAL,
                    fmt::format("Command '{}' did not produce expected file: {}", commandline(),
                                outfile.string()));
    }
  }
  return Status::OK;
}

std::string LocalShellCommand::commandline() const {
  std::string command;
  const std::vector<std::string>& args = arguments();
  for (size_t i = 0; i < args.size(); ++i) {
    if (i) {
      command += ' ';
    }
    command += args[i];
  }
  return command;
}

Status checkOutput(const std::string& cmd, int timeoutSeconds, const fs::path& workingDir,
                   std::string& stdout) {
  std::mutex m;
  std::condition_variable cv;
  Status returnStatus(StatusCode::INTERNAL, "Unknown error");

  std::thread t([&cmd, &workingDir, &cv, &returnStatus, &stdout]() {
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
        stdout = std::string(runOutput.first.buf.begin(), runOutput.first.buf.end());
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

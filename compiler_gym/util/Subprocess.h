// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>
#include <string>
#include <vector>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::util {

/**
 * A representation of a Command protocol buffer that describes a command that
 * is run in a subshell locally on the current machine.
 */
class LocalShellCommand {
 public:
  /**
   * Constructor.
   *
   * @param cmd The Command protocol buffer.
   */
  explicit LocalShellCommand(const Command& cmd);

  /**
   * Run the given command in a subshell.
   *
   * @return `OK` on success, `DEADLINE_EXCEEDED` on timeout, or `INTERNAL` if
   *    the command returns with a non-zero returncode.
   */
  grpc::Status checkCall() const;

  /**
   * Run the given command in a subshell and capture its stdout to string.
   *
   * @param stdout The string to set the stdout to.
   * @return `OK` on success, `DEADLINE_EXCEEDED` on timeout, or `INTERNAL` if
   *    the command returns with a non-zero returncode.
   */
  grpc::Status checkOutput(std::string& stdout) const;

  /**
   * Check that the specified infiles exist.
   *
   * @return `OK` on success, `INTERNAL` if a file is missing.
   */
  grpc::Status checkInfiles() const;

  /**
   * Check that the specified outfiles exist.
   *
   * @return `OK` on success, `INTERNAL` if a file is missing.
   */
  grpc::Status checkOutfiles() const;

  inline const std::vector<std::string>& arguments() const { return arguments_; };
  inline const std::chrono::seconds& timeout() const { return timeout_; }
  inline const int timeoutSeconds() const { return timeout_.count(); }
  inline const boost::process::environment& env() const { return env_; }
  inline const std::vector<boost::filesystem::path>& infiles() const { return infiles_; }
  inline const std::vector<boost::filesystem::path>& outfiles() const { return outfiles_; }

  /**
   * Get the list of command line arguments as a concatenated string.
   *
   * This is for debugging purposes, it should not be used to execute commands
   * as it does no escaping of arguments.
   */
  std::string commandline() const;

  /**
   * Returns whether this command instance has any arguments to execute.
   */
  const bool empty() const { return arguments_.empty(); }

  inline const Command proto() const { return proto_; };

 private:
  const Command proto_;
  const std::vector<std::string> arguments_;
  const std::chrono::seconds timeout_;
  const boost::process::environment env_;
  const std::vector<boost::filesystem::path> infiles_;
  const std::vector<boost::filesystem::path> outfiles_;
};

template <typename Rep, typename Period>
bool wait_for(boost::process::child& process, const std::chrono::duration<Rep, Period>& duration) {
#ifdef __APPLE__
  // FIXME(github.com/facebookresearch/CompilerGym/issues/399): Workaround for
  // an issue on macOS wherein boost::process:child::wait_for() blocks forever
  // at some (but not all) callsites. The problem does not occur on Linux, so
  // instead we just use the sans-timeout boost::process:child::wait() call and
  // rely on RPC call timeouts to handle the edge cases where commands time out.
  process.wait();
  return true;
#else  // Linux
  return process.wait_for(duration);
#endif
}

}  // namespace compiler_gym::util

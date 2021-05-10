// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <unistd.h>

#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/core/CompilerGymServicer.h"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/Unreachable.h"

DECLARE_string(port);
DECLARE_string(working_dir);

namespace compiler_gym {

// Increase maximum message size beyond the 4MB default as inbound message
// may be larger (e.g., in the case of IR strings).
constexpr size_t kMaxMessageSizeInBytes = 512 * 1024 * 1024;

// Create a service, configured using --port and --working_dir flags, and run
// it. This function never returns.
//
// CompilationService must be a valid compiler_gym::CompilationService subclass
// that implements the abstract methods and takes a single-argument working
// directory constructor:
//
//     class MyCompilationService final : public CompilationService {
//      public:
//       ...
//     }
//
// Usage:
//
//     int main(int argc, char** argv) {
//       createAndRunCompilerGymServiceImpl(argc, argv, "usage string");
//     }
template <typename CompilationSession>
[[noreturn]] void createAndRunCompilerGymServiceImpl(int argc, char** argv, const char* usage) {
  gflags::SetUsageMessage(std::string(usage));
  // TODO: Fatal error if unparsed flags remain.
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/false);

  // TODO: Create a temporary working directory if --working_dir is not set.
  CHECK(!FLAGS_working_dir.empty()) << "--working_dir flag not set";
  if (FLAGS_port.empty()) {
    FLAGS_port = "0";
  }

  const boost::filesystem::path workingDirectory = FLAGS_working_dir;
  FLAGS_log_dir = workingDirectory.string() + "/logs";

  CHECK(boost::filesystem::is_directory(FLAGS_log_dir)) << "Directory not found: " << FLAGS_log_dir;

  google::InitGoogleLogging(argv[0]);

  CompilerGymServicer<CompilationSession> service{workingDirectory};

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);

  builder.SetMaxMessageSize(kMaxMessageSizeInBytes);

  // Start a channel on the port.
  int port;
  std::string serverAddress = "0.0.0.0:" + FLAGS_port;
  builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials(), &port);

  // Start the server.
  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  CHECK(server) << "Failed to build RPC service";

  {
    // Write the port to a <working_dir>/port.txt file, which an external
    // process can read to determine how to get in touch. First write the port
    // to a temporary file and rename it, since renaming is atomic.
    const boost::filesystem::path portPath = workingDirectory / "port.txt";
    std::ofstream out(portPath.string() + ".tmp");
    out << std::to_string(port) << std::endl;
    out.close();
    boost::filesystem::rename(portPath.string() + ".tmp", portPath);
  }

  {
    // Write the process ID to a <working_dir>/pid.txt file, which can
    // external process can later use to determine if this service is still
    // alive.
    const boost::filesystem::path pidPath = workingDirectory / "pid.txt";
    std::ofstream out(pidPath.string() + ".tmp");
    out << std::to_string(getpid()) << std::endl;
    out.close();
    boost::filesystem::rename(pidPath.string() + ".tmp", pidPath);
  }

  LOG(INFO) << "Service " << workingDirectory << " listening on " << port << ", PID = " << getpid();

  server->Wait();
  UNREACHABLE("grpc::Server::Wait() should not return");
}

}  // namespace compiler_gym

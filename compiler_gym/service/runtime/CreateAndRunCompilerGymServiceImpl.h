// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//
// Private implementation of the createAndRunCompilerGymService(). Do not
// include this header directly! Use compiler_gym/service/runtime/Runtime.h.
#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <unistd.h>

#include <csignal>
#include <future>
#include <iostream>
#include <string>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/service/runtime/CompilerGymService.h"

DECLARE_string(port);
DECLARE_string(working_dir);

namespace compiler_gym::runtime {

extern std::promise<void> shutdownSignal;

void shutdown_handler(int signum);

void setGrpcChannelOptions(grpc::ServerBuilder& builder);

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
template <typename CompilationSessionType>
[[nodiscard]] int createAndRunCompilerGymServiceImpl(int argc, char** argv, const char* usage) {
  // Register a signal handler for SIGTERM that will set the shutdown_signal
  // future value.
  std::signal(SIGTERM, shutdown_handler);

  gflags::SetUsageMessage(std::string(usage));

  // Parse the command line arguments and die if any are unrecognized.
  gflags::ParseCommandLineFlags(&argc, &argv, /*remove_flags=*/true);
  if (argc > 1) {
    std::cerr << "ERROR: unknown command line argument '" << argv[1] << '\'';
    return 1;
  }

  // Set up the working and logging directories.
  boost::filesystem::path workingDirectory{FLAGS_working_dir};
  if (FLAGS_working_dir.empty()) {
    // If no working directory was set, create one.
    workingDirectory = boost::filesystem::unique_path(boost::filesystem::temp_directory_path() /
                                                      "compiler_gym-service-%%%%-%%%%");
    FLAGS_working_dir = workingDirectory.string();
  }

  // Create and set the logging directory.
  boost::filesystem::create_directories(workingDirectory / "logs");
  FLAGS_log_dir = workingDirectory.string() + "/logs";

  google::InitGoogleLogging(argv[0]);

  CompilerGymService<CompilationSessionType> service{workingDirectory};

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);

  setGrpcChannelOptions(builder);

  // Start a channel on the port.
  int port;
  std::string serverAddress = "0.0.0.0:" + (FLAGS_port.empty() ? "0" : FLAGS_port);
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

  // Block on the RPC service in a separate thread. This enables the current
  // thread to handle the shutdown routine.
  std::thread serverThread([&]() { server->Wait(); });

  // Block until this shutdown signal is received.
  shutdownSignal.get_future().wait();
  VLOG(2) << "Shutting down the RPC service";
  server->Shutdown();
  serverThread.join();
  VLOG(2) << "Service closed";

  if (service.sessionCount()) {
    LOG(ERROR) << "ERROR: Killing a service with " << service.sessionCount()
               << (service.sessionCount() > 1 ? " active sessions!" : " active session!")
               << std::endl;
    return 6;
  }

  return 0;
}

}  // namespace compiler_gym::runtime

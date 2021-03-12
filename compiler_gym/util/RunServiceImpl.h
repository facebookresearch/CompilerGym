// Private implementation header for //compiler_gym/util:RunService.
//
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <glog/logging.h>
#include <grpcpp/grpcpp.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <memory>
#include <string>

#include "boost/filesystem.hpp"

namespace compiler_gym::util {

// Create a service and run it. This function never returns.
template <typename Service>
int createAndRunService(const boost::filesystem::path& workingDirectory,
                        const std::string& requestedPort) {
  CHECK(boost::filesystem::is_directory(workingDirectory))
      << "Directory not found: " << workingDirectory.string();
  Service service{workingDirectory};

  grpc::ServerBuilder builder;
  builder.RegisterService(&service);

  // Increase maximum message size beyond the 4MB default as inbound message
  // may be larger (e.g., in the case of IR strings).
  builder.SetMaxMessageSize(512 * 1024 * 1024);

  // Start a channel on the port.
  int port;
  std::string serverAddress = "0.0.0.0:" + requestedPort;
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
    std::ofstream out(pidPath.string());
    out << std::to_string(getpid()) << std::endl;
    out.close();
  }

  LOG(INFO) << "Service " << workingDirectory << " listening on " << port << ", PID = " << getpid();

  server->Wait();
  return 0;
}

}  // namespace compiler_gym::util

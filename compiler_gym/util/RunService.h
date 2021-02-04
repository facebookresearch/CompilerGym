// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <unistd.h>

#include <string>

#include "compiler_gym/util/RunServiceImpl.h"

DECLARE_string(port);
DECLARE_string(working_dir);

namespace compiler_gym::util {

// Create a service, configured using --port and --working_dir flags, and
// run it. This function never returns.
//
// Service must be a subclass of CompilerGymService::Service that implements all
// RPC endpoints and takes a single-argument working directory constructor:
//
//    class MyService final : public CompilerGymService::Service {
//     public:
//      explicit MyService(const boost::filesystem::path& workingDirectory);
//    }
//
// Usage:
//
//     int main(int argc, char** argv) {
//         return runService<MyService>(&argc, &argv, "usage string");
//     }
template <typename Service>
int runService(int* argc, char*** argv, const char* usage) {
  gflags::SetUsageMessage(std::string(usage));
  gflags::ParseCommandLineFlags(argc, argv, /*remove_flags=*/false);

  CHECK(!FLAGS_working_dir.empty()) << "--working_dir flag not set";
  CHECK(!FLAGS_port.empty()) << "--port flag not set";

  FLAGS_log_dir = std::string(FLAGS_working_dir) + "/logs";
  google::InitGoogleLogging((*argv)[0]);

  return createAndRunService<Service>(FLAGS_working_dir, FLAGS_port);
}

}  // namespace compiler_gym::util

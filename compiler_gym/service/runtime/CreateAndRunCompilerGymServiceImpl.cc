// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/service/runtime/CreateAndRunCompilerGymServiceImpl.h"

DEFINE_string(
    working_dir, "",
    "The working directory to use. Must be an existing directory with write permissions.");
DEFINE_string(port, "0",
              "The port to listen on. If 0, an unused port will be selected. The selected port is "
              "written to <working_dir>/port.txt.");

namespace compiler_gym::runtime {

std::promise<void> shutdownSignal;

void shutdown_handler(int signum) {
  VLOG(1) << "Service received signal: " << signum;
  shutdownSignal.set_value();
}

// Configure the gRPC server, using the same options as the python client. See
// GRPC_CHANNEL_OPTIONS in compiler_gym/service/connection.py for the python
// equivalents and the rationale for each.
void setGrpcChannelOptions(grpc::ServerBuilder& builder) {
  builder.SetMaxReceiveMessageSize(-1);
  builder.SetMaxSendMessageSize(-1);
  builder.AddChannelArgument(GRPC_ARG_MAX_METADATA_SIZE, 512 * 1024);
  builder.AddChannelArgument(GRPC_ARG_ENABLE_HTTP_PROXY, 0);
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 0);
}

}  // namespace compiler_gym::runtime

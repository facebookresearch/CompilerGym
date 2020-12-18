// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/util/RunService.h"

DEFINE_string(
    working_dir, "",
    "The working directory to use. Must be an existing directory with write permissions.");
DEFINE_string(port, "0",
              "The port to listen on. If 0, an unused port will be selected. The selected port is "
              "written to <working_dir>/port.txt.");

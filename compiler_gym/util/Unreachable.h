// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <glog/logging.h>

// Declare a program point as unreachable. For debug builds, this will trigger
// a fatal error if reached. For optimized builds (i.e. ones built using
// `bazel build -c opt`), this is totally undefined.
#define UNREACHABLE(msg)                   \
  DLOG(FATAL) << "Unreachable: " << (msg); \
  __builtin_unreachable();

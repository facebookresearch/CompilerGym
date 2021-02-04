// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "glog/logging.h"
#include "include/grpcpp/grpcpp.h"

using grpc::Status;

#undef ASSERT_OK
#define ASSERT_OK(expr)                             \
  do {                                              \
    const Status _status = (expr);                  \
    CHECK(_status.ok()) << _status.error_message(); \
  } while (0)

#undef RETURN_IF_ERROR
#define RETURN_IF_ERROR(expr)      \
  do {                             \
    const Status _status = (expr); \
    if (!_status.ok())             \
      return _status;              \
  } while (0)

// Like RETURN_IF_ERROR(), but when you really want to commit!
#undef CRASH_IF_ERROR
#define CRASH_IF_ERROR(expr)                        \
  do {                                              \
    const Status _status = (expr);                  \
    CHECK(_status.ok()) << _status.error_message(); \
  } while (0)

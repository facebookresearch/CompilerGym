// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "glog/logging.h"
#include "include/grpcpp/grpcpp.h"

using grpc::Status;

#undef ASSERT_OK
/**
 * Fatal error if expression returns an error status.
 *
 * @param expr An expression that returns a `grpc::Status`.
 */
#define ASSERT_OK(expr)                             \
  do {                                              \
    const Status _status = (expr);                  \
    CHECK(_status.ok()) << _status.error_message(); \
  } while (0)

#undef RETURN_IF_ERROR
/**
 * Return from the current function if the expression returns an error status.
 *
 * This is equivalent to:
 *
 * \code{.cpp}
 *     Status status = expr;
 *     if (!status.ok()) {
 *         return status;
 *     }
 * \endcode
 *
 * @param expr An expression that return a `grpc::Status`.
 */
#define RETURN_IF_ERROR(expr)      \
  do {                             \
    const Status _status = (expr); \
    if (!_status.ok())             \
      return _status;              \
  } while (0)

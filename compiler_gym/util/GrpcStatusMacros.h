// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <fmt/format.h>

#include <boost/current_function.hpp>

#include "glog/logging.h"
#include "grpcpp/grpcpp.h"

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
 * Return from the current function if the expression returns an error status,
 * or if a std::exception is thrown.
 *
 * This is equivalent to:
 *
 *     \code{.cpp}
 *     try {
 *       Status status = expr;
 *       if (!status.ok()) {
 *         return status;
 *       }
 *     } catch (std::exception& e) {
 *       return E_STATUS;
 *     }
 *     \endcode
 *
 * @param expr An expression that return a `grpc::Status`.
 */
#define RETURN_IF_ERROR(expr)                                                                 \
  do {                                                                                        \
    try {                                                                                     \
      const Status _status = (expr);                                                          \
      if (!_status.ok())                                                                      \
        return _status;                                                                       \
    } catch (std::exception & e) {                                                            \
      return grpc::Status(grpc::StatusCode::INTERNAL,                                         \
                          fmt::format("Unhandled exception: {}\nSource: {}:{}\nFunction: {}", \
                                      e.what(), __FILE__, __LINE__, BOOST_CURRENT_FUNCTION)); \
    }                                                                                         \
  } while (0)

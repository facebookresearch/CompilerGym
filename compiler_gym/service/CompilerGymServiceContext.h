// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include "boost/filesystem.hpp"

namespace compiler_gym {

/**
 * Execution context of a compiler gym service.
 *
 * This class encapsulates mutable state that is shared between all compilation
 * sessions. An instance of this class is passed to every new
 * CompilationSession.
 *
 * You may subclass CompilerGymServiceContext to add additional mutable state.
 * The subclass .
 *
 * \code{.cpp}
 *
 *     #include "compiler_gym/service/CompilationSession.h"
 *     #include "compiler_gym/service/CompilerGymServiceContext.h"
 *     #include "compiler_gym/service/runtime/Runtime.h"
 *
 *     using namespace compiler_gym;
 *
 *     class MyServiceContext final : public CompilerGymServiceContext { ... }
 *
 *     class MyCompilationSession final : public CompilationSession { ... }
 *
 *     int main(int argc, char** argv) {
 *         return runtime::createAndRunCompilerGymService<MyCompilationSession, MyServiceContext>();
 *     }
 * \endcode
 */
class CompilerGymServiceContext {
 public:
  CompilerGymServiceContext(const boost::filesystem::path& workingDirectory);

  /**
   * Initialize context.
   *
   * Called before any compilation sessions are created. Use this method to
   * initialize any mutable state. If this routine returns an error, the service
   * will terminate.
   *
   * @return A status.
   */
  [[nodiscard]] virtual grpc::Status init();

  /**
   * Uninitialize context.
   *
   * Called after all compilation sessions have ended, before a service
   * terminates. Use this method to perform tidying up. This method is always
   * called, even if init() fails. If this routine returns an error, the service
   * will terminate with a nonzero error code.
   *
   * @return A status.
   */
  [[nodiscard]] virtual grpc::Status shutdown();

  /**
   * Get the working directory.
   *
   * The working directory is a local filesystem directory that compilation
   * sessions can use to store temporary files such as build artifacts. The
   * directory is guaranteed to exist.
   *
   * \note When possible, an in-memory filesystem will be used for the working
   *    directory. This means that the working directory is not suitable for
   *    very large files or executables, as some systems prevent execution of
   *    in-memory files.
   *
   * \note A single working directory is shared by all of the compilation
   * sessions of a service. Do not assume that you have exclusive access.
   *
   * @return A path.
   */
  inline const boost::filesystem::path& workingDirectory() const { return workingDirectory_; };

 private:
  const boost::filesystem::path workingDirectory_;
};

}  // namespace compiler_gym

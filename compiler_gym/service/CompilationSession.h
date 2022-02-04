// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <grpcpp/grpcpp.h>

#include <optional>
#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym {

/**
 * Base class for encapsulating an incremental compilation session.
 *
 * To add support for a new compiler, subclass from this base and provide
 * implementations of the abstract methods, then call
 * createAndRunCompilerGymService() and parametrize it with your class type:
 *
 * \code{.cpp}
 *     #include "compiler_gym/service/CompilationSession.h"
 *     #include "compiler_gym/service/runtime/Runtime.h"
 *
 *     using namespace compiler_gym;
 *
 *     class MyCompilationSession final : public CompilationSession { ... }
 *
 *     int main(int argc, char** argv) {
 *         runtime::createAndRunCompilerGymService<MyCompilationSession>();
 *     }
 * \endcode
 */
class CompilationSession {
 public:
  /**
   * Get the compiler version.
   *
   * @return A string indicating the compiler version.
   */
  virtual std::string getCompilerVersion() const;

  /**
   * A list of action spaces describing the capabilities of the compiler.
   *
   * @return A list of ActionSpace instances.
   */
  virtual std::vector<ActionSpace> getActionSpaces() const = 0;

  /**
   * A list of feature vectors that this compiler provides.
   *
   * @return A list of ObservationSpace instances.
   */
  virtual std::vector<ObservationSpace> getObservationSpaces() const = 0;

  /**
   * Start a CompilationSession.
   *
   * This will be called after construction and before applyAction() or
   * computeObservation(). This will only be called once.
   *
   * @param actionSpace The action space to use.
   * @param benchmark The benchmark to use.
   * @return `OK` on success, else an error code and message.
   */
  [[nodiscard]] virtual grpc::Status init(const ActionSpace& actionSpace,
                                          const Benchmark& benchmark) = 0;

  /**
   * Initialize a CompilationSession from another CompilerSession.
   *
   * Think of this like a copy constructor, except that this method is allowed
   * to fail.
   *
   * This will be called after construction and before applyAction() or
   * computeObservation(). This will only be called once.
   *
   * @param other The CompilationSession to initialize from.
   * @return `OK` on success, else an errro code and message.
   */
  [[nodiscard]] virtual grpc::Status init(CompilationSession* other);

  /**
   * Apply an action.
   *
   * @param action The action to apply.
   * @param newActionSpace If applying the action mutated the action space, set
   *    this value to the new action space.
   * @param actionHadNoEffect If the action had no effect, set this to true.
   * @return `OK` on success, else an errro code and message.
   */
  [[nodiscard]] virtual grpc::Status applyAction(const Event& action, bool& endOfEpisode,
                                                 std::optional<ActionSpace>& newActionSpace,
                                                 bool& actionHadNoEffect) = 0;

  /**
   * Compute an observation.
   *
   * @return `OK` on success, else an errro code and message.
   */
  [[nodiscard]] virtual grpc::Status computeObservation(const ObservationSpace& observationSpace,
                                                        Event& observation) = 0;

  /**
   * Optional. This will be called after all applyAction() and
   * computeObservation() in a step. Use this method if you would like to
   * perform post-transform validation of compiler state.
   *
   * @return `OK` on success, else an errro code and message.
   */
  [[nodiscard]] virtual grpc::Status endOfStep(bool actionHadNoEffect, bool& endOfEpisode,
                                               std::optional<ActionSpace>& newActionSpace);

  CompilationSession(const boost::filesystem::path& workingDirectory);

  virtual ~CompilationSession() = default;

  /**
   * Handle a session parameter send by the frontend.
   *
   * Session parameters provide a method to send ad-hoc key-value messages to a
   * compilation session through the env.send_session_parameter() method. It us
   * up to the client/service to agree on a common schema for encoding and
   * decoding these parameters.
   *
   * Implementing this method is optional.
   *
   * @param key The parameter key.
   * @param value The parameter value.
   * @param reply A string response message for the parameter, or leave as
   *    std::nullopt if the parameter is unknown.
   * @return `OK` on success, else an errro code and message.
   */
  [[nodiscard]] virtual grpc::Status handleSessionParameter(const std::string& key,
                                                            const std::string& value,
                                                            std::optional<std::string>& reply);

 protected:
  /**
   * Get the working directory.
   *
   * The working directory is a local filesystem directory that this
   * CompilationSession can use to store temporary files such as build
   * artifacts. The directory exists.
   *
   * \note If you need to store very large files for a CompilationSession then
   *    consider using an alternate filesystem path as, when possible, an
   *    in-memory filesystem will be used for the working directory.
   *
   * \note A single working directory may be shared by multiple
   *    CompilationSession instances. Do not assume that you have exclusive
   *    access.
   *
   * @return A path.
   */
  inline const boost::filesystem::path& workingDirectory() { return workingDirectory_; }

 private:
  const boost::filesystem::path workingDirectory_;
};

}  // namespace compiler_gym

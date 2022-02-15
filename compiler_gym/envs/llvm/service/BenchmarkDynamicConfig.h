// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <vector>

#include "boost/filesystem.hpp"
#include "compiler_gym/service/proto/compiler_gym_service.pb.h"
#include "compiler_gym/util/Subprocess.h"

namespace compiler_gym::llvm_service {

/**
 * Represents a BenchmarkDynamicConfig protocol buffer.
 */
class BenchmarkDynamicConfig {
 public:
  explicit BenchmarkDynamicConfig(const compiler_gym::BenchmarkDynamicConfig& cfg);

  inline const util::LocalShellCommand& buildCommand() const { return buildCommand_; };
  inline const util::LocalShellCommand& runCommand() const { return runCommand_; };
  inline const std::vector<util::LocalShellCommand>& preRunCommands() const {
    return preRunCommands_;
  };
  inline const std::vector<util::LocalShellCommand>& postRunCommands() const {
    return postRunCommands_;
  };
  inline bool isBuildable() const { return isBuildable_; }
  inline bool isRunnable() const { return isRunnable_; }

 private:
  const util::LocalShellCommand buildCommand_;
  const util::LocalShellCommand runCommand_;
  const std::vector<util::LocalShellCommand> preRunCommands_;
  const std::vector<util::LocalShellCommand> postRunCommands_;
  const bool isBuildable_;
  const bool isRunnable_;
};

BenchmarkDynamicConfig realizeDynamicConfig(const compiler_gym::BenchmarkDynamicConfig& proto,
                                            const boost::filesystem::path& scratchDirectory);

}  // namespace compiler_gym::llvm_service

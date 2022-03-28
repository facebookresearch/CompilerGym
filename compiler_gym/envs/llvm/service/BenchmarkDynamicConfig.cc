// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/BenchmarkDynamicConfig.h"

#include "compiler_gym/util/RunfilesPath.h"

namespace fs = boost::filesystem;

using BenchmarkDynamicConfigProto = compiler_gym::BenchmarkDynamicConfig;

namespace compiler_gym::llvm_service {

namespace {

std::vector<util::LocalShellCommand> commandsFromProto(
    const google::protobuf::RepeatedPtrField<Command>& cmds) {
  std::vector<util::LocalShellCommand> outs;
  for (const auto& cmd : cmds) {
    outs.push_back(util::LocalShellCommand(cmd));
  }
  return outs;
}

}  // anonymous namespace

BenchmarkDynamicConfig realizeDynamicConfig(const BenchmarkDynamicConfigProto& proto,
                                            const fs::path& scratchDirectory) {
  compiler_gym::BenchmarkDynamicConfig cfg;
  cfg.CopyFrom(proto);

  // Set up the environment variables.
  (*cfg.mutable_build_cmd()->mutable_env())["CC"] =
      util::getSiteDataPath("llvm-v0/bin/clang").string();
  (*cfg.mutable_build_cmd()->mutable_env())["IN"] = (scratchDirectory / "out.bc").string();

  // Register the IR as a pre-requisite build file.
  cfg.mutable_build_cmd()->add_infile((scratchDirectory / "out.bc").string());

  return BenchmarkDynamicConfig(cfg, scratchDirectory);
}

BenchmarkDynamicConfig::BenchmarkDynamicConfig(const BenchmarkDynamicConfigProto& cfg,
                                               const boost::filesystem::path& scratchDirectory)
    : buildCommand_(cfg.build_cmd()),
      runCommand_(cfg.run_cmd()),
      preRunCommands_(commandsFromProto(cfg.pre_run_cmd())),
      postRunCommands_(commandsFromProto(cfg.post_run_cmd())),
      isBuildable_(!buildCommand_.empty()),
      isRunnable_(!(buildCommand_.empty() || runCommand_.empty())),
      scratchDirectory_(scratchDirectory) {}

}  // namespace compiler_gym::llvm_service

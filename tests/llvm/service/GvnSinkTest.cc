// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <gtest/gtest.h>

#include <boost/filesystem.hpp>
#include <optional>

#include "compiler_gym/envs/llvm/service/Benchmark.h"
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"
#include "compiler_gym/envs/llvm/service/LlvmSession.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "glog/logging.h"

using namespace ::testing;

namespace fs = boost::filesystem;

namespace compiler_gym::llvm_service {
namespace {

class GvnSinkTest : public ::testing::Test {
 public:
  void SetUp() {
    workingDirectory_ = fs::temp_directory_path() / fs::unique_path();
    fs::create_directory(workingDirectory_);
  }

  void TearDown() { fs::remove_all(workingDirectory_); }

 protected:
  boost::filesystem::path workingDirectory_;
};

TEST_F(GvnSinkTest, runGvnSinkOnBlowfish) {
  const auto blowfish =
      util::getRunfilesPath("compiler_gym/third_party/cbench/cbench-v1/blowfish.bc");

  BenchmarkFactory factory(workingDirectory_);
  ASSERT_OK(factory.addBitcodeFile("benchmark://cbench-v1/blowfish", blowfish));
  std::unique_ptr<Benchmark> benchmark;
  ASSERT_OK(factory.getBenchmark("benchmark://cbench-v1/blowfish", &benchmark));

  LlvmSession env(std::move(benchmark), LlvmActionSpace::PASSES_ALL, LlvmObservationSpace::IR,
                  std::nullopt, workingDirectory_);

  StepRequest request;
  request.add_action(static_cast<int>(LlvmAction::GVNSINK_PASS));
  StepReply reply;
  ASSERT_OK(env.Step(request, &reply));
}

}  // anonymous namespace
}  // namespace compiler_gym::llvm_service

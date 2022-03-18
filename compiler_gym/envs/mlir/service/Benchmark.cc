// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/mlir/service/Benchmark.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <chrono>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "compiler_gym/envs/mlir/service/MlirUtils.h"
#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/Subprocess.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVMPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"

namespace fs = boost::filesystem;
namespace sys = boost::system;

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym::mlir_service {

namespace {

std::unique_ptr<mlir::OwningModuleRef> makeModuleOrDie(mlir::MLIRContext& context,
                                                       const Bitcode& bitcode,
                                                       const std::string& name) {
  Status status;
  auto module = makeModule(context, bitcode, name, &status);
  CHECK(status.ok()) << "Failed to make MLIR module: " << status.error_message();
  return module;
}

RealizedBenchmarkDynamicConfig realizeDynamicConfig(const BenchmarkDynamicConfig& original,
                                                    const fs::path& scratchDirectory) {
  BenchmarkDynamicConfig cfg;
  cfg.CopyFrom(original);

  auto command = cfg.mutable_build_cmd();

  // TODO(boian): clang and Benchmark must be included in the compiler_gym python package.
  // Take paths in a common way.
  const auto llvm_install = util::getRunfilesPath("external/llvm-13/install/");
  const auto benchmark_install = util::getRunfilesPath("external/benchmark/install/");

  // Set up the environment variables.
  (*command->mutable_env())["CC"] = (llvm_install / "bin/clang").string();
  // Register the IR as a pre-requisite build file.
  command->add_infile((scratchDirectory / "out.mlir.ll").string());
  command->add_infile((scratchDirectory / "benchmark_main.cc").string());

  command->mutable_argument()->Clear();
  command->add_argument("$CC");
  command->add_argument("-Ofast");
  command->add_argument((scratchDirectory / "out.mlir.ll").string());
  command->add_argument("-mllvm");
  command->add_argument("-enable-matrix");
  command->add_argument("-mllvm");
  command->add_argument("-matrix-allow-contract");
  command->add_argument("-mllvm");
  command->add_argument("-matrix-default-layout=row-major");
  command->add_argument("-c");
  command->add_argument("-o");
  command->add_argument((scratchDirectory / "benchmark_mlir.o").string());
  command->add_argument((llvm_install / "lib/libMLIRExecutionEngine.a").string());
  command->add_argument("-g3");
  command->add_argument("&&");

  // TODO(boian): Move out the compilation of the main object file
  // to when the benchmark is initialized.
  // Ideally, the main executable should not need relinking.
  // The MLIR code should be compiled into a shared lib and then dynamically loaded.
  // These optimization should reduce the overhead.

  // Compile and link
  command->add_argument("$CC");
  command->add_argument("-o");
  command->add_argument((scratchDirectory / "benchmark_main").string());
  command->add_argument("-I" + (llvm_install / "include/").string());
  command->add_argument("-I" + (benchmark_install / "include/").string());
  command->add_argument("-lstdc++");
  command->add_argument("-lm");
  command->add_argument("-pthread");
  command->add_argument("-v");
  command->add_argument("-g3");
  command->add_argument((scratchDirectory / "benchmark_main.cc").string());
  command->add_argument((scratchDirectory / "benchmark_mlir.o").string());
  command->add_argument((benchmark_install / "lib/libbenchmark.a").string());

  return RealizedBenchmarkDynamicConfig(cfg);
}

}  // anonymous namespace

Status readBitcodeFile(const fs::path& path, Bitcode* bitcode) {
  std::ifstream ifs(path.string());
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("File not found: \"{}\"", path.string()));
  }

  ifs.seekg(0, std::ios::end);
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Error reading file: \"{}\"", path.string()));
  }

  std::streampos fileSize = ifs.tellg();
  if (!fileSize) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("File is empty: \"{}\"", path.string()));
  }

  bitcode->resize(fileSize);
  ifs.seekg(0);
  ifs.read(&(*bitcode)[0], bitcode->size());
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Error reading file: \"{}\"", path.string()));
  }

  return Status::OK;
}

std::vector<int> parametersFromUri(std::string uri) {
  std::vector<int> ret;
  std::vector<std::string> tokensSlash;
  boost::split(tokensSlash, uri, boost::is_any_of("/"));
  std::vector<std::string> tokensUnderscore;
  boost::split(tokensUnderscore, tokensSlash[tokensSlash.size() - 1], boost::is_any_of("_"));
  for (auto i = 0; i < tokensUnderscore.size(); i++) {
    ret.push_back(std::stoi(tokensUnderscore[i]));
  }
  return ret;
}

std::unique_ptr<mlir::OwningModuleRef> makeModule(mlir::MLIRContext& context,
                                                  const Bitcode& bitcode, const std::string& name,
                                                  Status* status) {
  auto params = parametersFromUri(name);
  int m, n, k;
  m = params[0];
  n = params[1];
  k = params[2];
  Bitcode mlirSource = bitcode;
  boost::replace_all(mlirSource, "${M}", std::to_string(m));
  boost::replace_all(mlirSource, "${N}", std::to_string(n));
  boost::replace_all(mlirSource, "${K}", std::to_string(k));
  mlir::OwningModuleRef moduleRef = parseSourceString(bitcode, &context);
  if (!moduleRef) {
    *status = Status(StatusCode::INVALID_ARGUMENT,
                     fmt::format("Failed to parse MLIR bitcode: \"{}\" :\n {}", name, bitcode));
    return nullptr;
  }

  mlir::ModuleOp module = *moduleRef;
  // Strip the module identifiers and source file names from the module to
  // anonymize them. This is to deter learning algorithms from overfitting to
  // benchmarks by their name.
  module.setName("-");

  return std::make_unique<mlir::OwningModuleRef>(std::move(moduleRef));
}

// A benchmark is an MLIR module and the MLIR context that owns it.
Benchmark::Benchmark(const std::string& name, const Bitcode& bitcode,
                     const BenchmarkDynamicConfig& dynamicConfig, const fs::path& workingDirectory)
    : scratchDirectory_(fs::path(fs::unique_path(workingDirectory / "scratch-%%%%-%%%%"))),
      dynamicConfigProto_(dynamicConfig),
      dynamicConfig_(realizeDynamicConfig(dynamicConfig, scratchDirectory_)),
      name_(name),
      needsRecompile_(true),
      runtimesPerObservationCount_(kDefaultRuntimesPerObservationCount),
      warmupRunsPerRuntimeObservationCount_(kDefaultWarmupRunsPerRuntimeObservationCount),
      buildtimesPerObservationCount_(kDefaultBuildtimesPerObservationCount) {
  context_ = mlir::createMlirContext();

  module_ = (makeModuleOrDie(*context_, bitcode, name));
  auto params = parametersFromUri(name);
  m_ = params[0];
  n_ = params[1];
  k_ = params[2];

  sys::error_code ec;
  fs::create_directory(scratchDirectory(), ec);
  CHECK(!ec) << "Failed to create scratch directory: " << scratchDirectory();
}

Benchmark::Benchmark(const std::string& name, std::unique_ptr<mlir::MLIRContext> context,
                     std::unique_ptr<mlir::OwningModuleRef> module,
                     const BenchmarkDynamicConfig& dynamicConfig, const fs::path& workingDirectory)
    : context_(std::move(context)),
      module_(std::move(module)),
      scratchDirectory_(fs::path(fs::unique_path(workingDirectory / "scratch-%%%%-%%%%"))),
      dynamicConfigProto_(dynamicConfig),
      dynamicConfig_(realizeDynamicConfig(dynamicConfig, scratchDirectory_)),
      name_(name),
      needsRecompile_(true) {
  auto params = parametersFromUri(name);
  m_ = params[0];
  n_ = params[1];
  k_ = params[2];

  sys::error_code ec;
  fs::create_directory(scratchDirectory(), ec);
  CHECK(!ec) << "Failed to create scratch directory: " << scratchDirectory();
}

std::unique_ptr<Benchmark> Benchmark::clone(const fs::path& workingDirectory) {
  std::string bitcode;
  llvm::raw_string_ostream ss(bitcode);

  module()->print(ss);

  return std::make_unique<Benchmark>(name(), bitcode, dynamicConfigProto_, workingDirectory);
}

Status Benchmark::verify_module() {
  std::string errorMessage;
  // TODO(kyleherndon)
  // llvm::raw_string_ostream rso(errorMessage);
  // if (llvm::verifyModule(module(), &rso)) {
  //   rso.flush();
  //   return Status(StatusCode::DATA_LOSS, "Failed to verify module: " + errorMessage);
  // }
  return Status::OK;
}

Status writeBitcodeFile(mlir::OwningModuleRef& module, const fs::path& path) {
  std::ofstream output(path);

  std::string bitcode;
  llvm::raw_string_ostream ss(bitcode);

  module->print(ss);

  output << bitcode;
  output.close();

  if (!output) {
    return Status(StatusCode::PERMISSION_DENIED,
                  "Failed to write bitcode to file: " + path.string());
  }

  return Status::OK;
}

Status Benchmark::writeBitcodeToFile(const fs::path& path) {
  return writeBitcodeFile(module(), path);
}

Status Benchmark::computeRuntime(Event& observation) {
  const RealizedBenchmarkDynamicConfig& cfg = dynamicConfig();

  if (!cfg.isRunnable()) {
    return Status::OK;
  }

  if (chdir(scratchDirectory().string().c_str())) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to set working directory: {}", scratchDirectory().string()));
  }

  RETURN_IF_ERROR(compile());

  // Run the pre-execution hooks.
  for (const auto& preRunCommand : cfg.preRunCommands()) {
    RETURN_IF_ERROR(preRunCommand.checkInfiles());
    RETURN_IF_ERROR(preRunCommand.checkCall());
    RETURN_IF_ERROR(preRunCommand.checkOutfiles());
  }

  RETURN_IF_ERROR(cfg.runCommand().checkInfiles());

  // Run the binary.
  VLOG(3) << "Running " << getRuntimesPerObservationCount() << " iterations of binary";
  *observation.mutable_double_tensor()->mutable_shape()->Add() = getRuntimesPerObservationCount();
  for (int i = 0; i < getRuntimesPerObservationCount(); ++i) {
    std::string result;
    RETURN_IF_ERROR(cfg.runCommand().checkOutput(result));
    nlohmann::json result_json;
    std::stringstream result_stream(result);
    result_stream >> result_json;
    *observation.mutable_double_tensor()->mutable_value()->Add() =
        result_json.at("benchmarks").at(0).at("cpu_time").get<double>() / 1000000000;
  }

  RETURN_IF_ERROR(cfg.runCommand().checkOutfiles());

  // Run the post-execution hooks.
  for (const auto& postRunCommand : cfg.postRunCommands()) {
    RETURN_IF_ERROR(postRunCommand.checkInfiles());
    RETURN_IF_ERROR(postRunCommand.checkCall());
    RETURN_IF_ERROR(postRunCommand.checkOutfiles());
  }

  return Status::OK;
}

Status Benchmark::computeBuildtime(Event& observation) {
  if (!dynamicConfig().isBuildable()) {
    return Status::OK;
  }

  RETURN_IF_ERROR(compile());

  *observation.mutable_double_tensor()->mutable_shape()->Add() = 1;
  *observation.mutable_double_tensor()->mutable_value()->Add() =
      static_cast<double>(lastBuildTimeMicroseconds()) / 1000000;

  return Status::OK;
}

Status Benchmark::compile() {
  const auto& cfg = dynamicConfig();

  if (!cfg.isBuildable()) {
    return Status::OK;
  }

  if (!needsRecompile_) {
    return Status::OK;
  }

  VLOG(3) << "Compiling benchmark";

  if (chdir(scratchDirectory().string().c_str())) {
    return Status(StatusCode::INTERNAL,
                  fmt::format("Failed to set working directory: {}", scratchDirectory().string()));
  }

  std::string bitcode;
  llvm::raw_string_ostream ss(bitcode);

  RETURN_IF_ERROR(lowerMLIRModuleToLLVM(*module_, context_.get(), ss));
  // return Status(StatusCode::INTERNAL, bitcode);

  std::ofstream mlir_file(scratchDirectory() / "out.mlir.ll");
  mlir_file << bitcode;
  mlir_file.close();
  if (!mlir_file) {
    return Status(StatusCode::PERMISSION_DENIED,
                  "Failed to write mlir file: " + (scratchDirectory() / "out.mlir.ll").string());
  }

  auto mainTemplatePath = util::getSiteDataPath("mlir-v0/benchmark_main.cc.template");
  std::ifstream t(mainTemplatePath);
  std::stringstream buffer;
  buffer << t.rdbuf();
  std::string main = buffer.str();
  boost::replace_all(main, "${M}", std::to_string(m_));
  boost::replace_all(main, "${N}", std::to_string(n_));
  boost::replace_all(main, "${K}", std::to_string(k_));
  std::ofstream main_file(scratchDirectory() / "benchmark_main.cc");
  main_file << main;
  main_file.close();
  if (!main_file) {
    return Status(
        StatusCode::PERMISSION_DENIED,
        "Failed to write main file: " + (scratchDirectory() / "benchmark_main.cc").string());
  }

  // Check that the required sources exist.
  RETURN_IF_ERROR(cfg.buildCommand().checkInfiles());

  // Build the bitcode.
  const std::chrono::time_point<std::chrono::steady_clock> start = std::chrono::steady_clock::now();
  RETURN_IF_ERROR(cfg.buildCommand().checkCall());

  const auto end = std::chrono::steady_clock::now();

  // Check that the expected output files were generated.
  RETURN_IF_ERROR(cfg.buildCommand().checkOutfiles());

  buildTimeMicroseconds_ =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  needsRecompile_ = false;
  return Status::OK;
}

// bool Benchmark::applyBaselineOptimizations(unsigned optLevel, unsigned sizeLevel) {
//   return applyBaselineOptimizationsToModule(&module(), optLevel, sizeLevel);
// }

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

RealizedBenchmarkDynamicConfig::RealizedBenchmarkDynamicConfig(const BenchmarkDynamicConfig& cfg)
    : buildCommand_(cfg.build_cmd()),
      runCommand_(cfg.run_cmd()),
      preRunCommands_(commandsFromProto(cfg.pre_run_cmd())),
      postRunCommands_(commandsFromProto(cfg.post_run_cmd())),
      isBuildable_(!buildCommand_.empty()),
      isRunnable_(!(buildCommand_.empty() || runCommand_.empty())) {}

}  // namespace compiler_gym::mlir_service

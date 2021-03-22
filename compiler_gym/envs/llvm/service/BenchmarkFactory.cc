// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"

#include <fmt/format.h>
#include <glog/logging.h>

#include <iostream>
#include <memory>
#include <string>

#include "compiler_gym/util/GrpcStatusMacros.h"
#include "compiler_gym/util/RunfilesPath.h"
#include "compiler_gym/util/StrLenConstexpr.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

namespace fs = boost::filesystem;

using grpc::Status;
using grpc::StatusCode;

namespace compiler_gym::llvm_service {

static const std::string kExpectedExtension = ".bc";

static const fs::path kSiteBenchmarksDir = util::getSiteDataPath("llvm/10.0.0/bitcode_benchmarks");

BenchmarkFactory::BenchmarkFactory(const boost::filesystem::path& workingDirectory,
                                   std::optional<std::mt19937_64> rand,
                                   size_t maxLoadedBenchmarkSize)
    : workingDirectory_(workingDirectory),
      rand_(rand.has_value() ? *rand : std::mt19937_64(std::random_device()())),
      loadedBenchmarksSize_(0),
      maxLoadedBenchmarkSize_(maxLoadedBenchmarkSize) {
  // Register all benchmarks from the site data directory.
  if (fs::is_directory(kSiteBenchmarksDir)) {
    CRASH_IF_ERROR(scanSiteDataDirectory());
  } else {
    LOG(INFO) << "LLVM site benchmark directory not found: " << kSiteBenchmarksDir.string();
  }

  VLOG(2) << "BenchmarkFactory initialized with " << numBenchmarks() << " benchmarks";
}

Status BenchmarkFactory::addBitcode(const std::string& uri, const Bitcode& bitcode,
                                    std::optional<fs::path> bitcodePath) {
  Status status;
  std::unique_ptr<llvm::LLVMContext> context = std::make_unique<llvm::LLVMContext>();
  std::unique_ptr<llvm::Module> module = makeModule(*context, bitcode, uri, &status);
  RETURN_IF_ERROR(status);
  DCHECK(module);

  const size_t bitcodeSize = bitcode.size();
  if (loadedBenchmarksSize_ + bitcodeSize > maxLoadedBenchmarkSize_) {
    VLOG(2) << "Adding new bitcode with size " << bitcodeSize
            << " exceeds maximum in-memory cache capacity " << maxLoadedBenchmarkSize_ << ", "
            << benchmarks_.size() << " bitcodes";
    int evicted = 0;
    // Evict benchmarks until we have reduced capacity below 50%. Use a
    // bounded for loop to prevent infinite loop if we get "unlucky" and
    // have no valid candidates to unload.
    const size_t targetCapacity = maxLoadedBenchmarkSize_ / 2;
    for (size_t i = 0; i < benchmarks_.size() * 2; ++i) {
      // We have run out of benchmarks to evict, or have freed up
      // enough capacity.
      if (!benchmarks_.size() || loadedBenchmarksSize_ < targetCapacity) {
        break;
      }

      // Select a cached benchmark randomly.
      std::uniform_int_distribution<size_t> distribution(0, benchmarks_.size() - 1);
      size_t index = distribution(rand_);
      auto iterator = std::next(std::begin(benchmarks_), index);

      // Check that the benchmark has an on-disk bitcode file which
      // can be loaded to re-cache this bitcode. If not, we cannot
      // evict it.
      if (!iterator->second.bitcodePath().has_value()) {
        continue;
      }

      // Evict the benchmark: add it to the pool of unloaded benchmarks and
      // delete it from the pool of loaded benchmarks.
      ++evicted;
      loadedBenchmarksSize_ -= iterator->second.bitcodeSize();
      unloadedBitcodePaths_.insert({iterator->first, *iterator->second.bitcodePath()});
      benchmarks_.erase(iterator);
    }

    VLOG(3) << "Evicted " << evicted << " benchmarks. Bitcode cache size now "
            << loadedBenchmarksSize_ << ", " << benchmarks_.size() << " bitcodes";
  }

  benchmarks_.insert({uri, Benchmark(uri, std::move(context), std::move(module), bitcodeSize,
                                     workingDirectory_, bitcodePath)});
  loadedBenchmarksSize_ += bitcodeSize;

  return Status::OK;
}

Status BenchmarkFactory::addBitcodeFile(const std::string& uri,
                                        const boost::filesystem::path& path) {
  if (!fs::exists(path)) {
    return Status(StatusCode::NOT_FOUND, fmt::format("File not found: \"{}\"", path.string()));
  }
  unloadedBitcodePaths_[uri] = path;
  return Status::OK;
}

Status BenchmarkFactory::addBitcodeUriAlias(const std::string& src, const std::string& dst) {
  // TODO(github.com/facebookresearch/CompilerGym/issues/2): Add support
  // for additional protocols, e.g. http://.
  if (dst.rfind("file:////", 0) != 0) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("Unsupported benchmark URI protocol: \"{}\"", dst));
  }

  // Resolve path from file:/// protocol URI.
  const boost::filesystem::path path{dst.substr(util::strLen("file:///"))};
  return addBitcodeFile(src, path);
}

namespace {

bool endsWith(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // anonymous namespace

Status BenchmarkFactory::addDirectoryOfBitcodes(const boost::filesystem::path& root) {
  VLOG(3) << "addDirectoryOfBitcodes(" << root.string() << ")";
  if (!fs::is_directory(root)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  fmt::format("Directory not found: \"{}\"", root.string()));
  }

  // Check if there is a manifest file that we can read, rather than having to
  // enumerate the directory ourselves.
  const auto manifestPath = fs::path(root.string() + ".MANIFEST");
  if (fs::is_regular_file(manifestPath)) {
    VLOG(3) << "Reading manifest file: " << manifestPath;
    return addDirectoryOfBitcodes(root, manifestPath);
  }

  const auto rootPathSize = root.string().size();
  for (auto it : fs::recursive_directory_iterator(root, fs::symlink_option::recurse)) {
    if (!fs::is_regular_file(it)) {
      continue;
    }
    const std::string& path = it.path().string();

    if (!endsWith(path, kExpectedExtension)) {
      continue;
    }

    // The name of the benchmark is path, relative to the root, without the
    // file extension.
    const std::string name =
        path.substr(rootPathSize + 1, path.size() - rootPathSize - kExpectedExtension.size() - 1);
    const std::string uri = fmt::format("benchmark://{}", name);

    RETURN_IF_ERROR(addBitcodeFile(uri, path));
  }

  return Status::OK;
}

Status BenchmarkFactory::addDirectoryOfBitcodes(const boost::filesystem::path& root,
                                                const boost::filesystem::path& manifestPath) {
  std::ifstream infile(manifestPath.string());
  std::string relPath;
  while (std::getline(infile, relPath)) {
    if (!endsWith(relPath, kExpectedExtension)) {
      continue;
    }

    const fs::path path = root / relPath;
    const std::string name = relPath.substr(0, relPath.size() - kExpectedExtension.size());
    const std::string uri = fmt::format("benchmark://{}", name);

    RETURN_IF_ERROR(addBitcodeFile(uri, path));
  }

  return Status::OK;
}

Status BenchmarkFactory::getBenchmark(std::unique_ptr<Benchmark>* benchmark) {
  if (!benchmarks_.size() && !unloadedBitcodePaths_.size()) {
    return Status(StatusCode::NOT_FOUND,
                  fmt::format("No benchmarks registered. Site data directory: `{}`",
                              kSiteBenchmarksDir.string()));
  }

  const size_t unloadedBenchmarkCount = unloadedBitcodePaths_.size();
  const size_t loadedBenchmarkCount = benchmarks_.size();

  const size_t benchmarkCount = unloadedBenchmarkCount + loadedBenchmarkCount;

  std::uniform_int_distribution<size_t> distribution(0, benchmarkCount - 1);
  size_t index = distribution(rand_);

  if (index < unloadedBenchmarkCount) {
    // Select a random unloaded benchmark to load and move to the loaded
    // benchmark collection.
    auto unloadedBenchmark = std::next(std::begin(unloadedBitcodePaths_), index);
    CHECK(unloadedBenchmark != unloadedBitcodePaths_.end());
    RETURN_IF_ERROR(loadBenchmark(unloadedBenchmark, benchmark));
  } else {
    auto loadedBenchmark = std::next(std::begin(benchmarks_), index - unloadedBenchmarkCount);
    CHECK(loadedBenchmark != benchmarks_.end());
    *benchmark = loadedBenchmark->second.clone(workingDirectory_);
  }

  return Status::OK;
}

Status BenchmarkFactory::getBenchmark(const std::string& uri,
                                      std::unique_ptr<Benchmark>* benchmark) {
  std::string resolvedUri = uri;
  // Prepend benchmark:// protocol if not specified. E.g. "foo/bar" resolves to
  // "benchmark://foo/bar", but "file:///foo/bar" is not changed.
  if (uri.find("://") == std::string::npos) {
    resolvedUri = fmt::format("benchmark://{}", uri);
  }

  auto loaded = benchmarks_.find(resolvedUri);
  if (loaded != benchmarks_.end()) {
    *benchmark = loaded->second.clone(workingDirectory_);
    return Status::OK;
  }

  auto unloaded = unloadedBitcodePaths_.find(resolvedUri);
  if (unloaded != unloadedBitcodePaths_.end()) {
    RETURN_IF_ERROR(loadBenchmark(unloaded, benchmark));
    return Status::OK;
  }

  // No exact name match - attempt to match the URI prefix.
  return getBenchmarkByUriPrefix(uri, resolvedUri, benchmark);
}

Status BenchmarkFactory::getBenchmarkByUriPrefix(const std::string& uriPrefix,
                                                 const std::string& resolvedUriPrefix,
                                                 std::unique_ptr<Benchmark>* benchmark) {
  // Make a list of all of the known benchmarks which match this prefix.
  std::vector<const char*> candidateBenchmarks;
  for (const auto& it : unloadedBitcodePaths_) {
    const std::string& uri = it.first;
    if (uri.rfind(resolvedUriPrefix, 0) == 0) {
      candidateBenchmarks.push_back(uri.c_str());
    }
  }
  for (const auto& it : benchmarks_) {
    const std::string& uri = it.first;
    if (uri.rfind(resolvedUriPrefix, 0) == 0) {
      candidateBenchmarks.push_back(uri.c_str());
    }
  }

  const size_t candidatesCount = candidateBenchmarks.size();
  if (!candidatesCount) {
    return Status(StatusCode::INVALID_ARGUMENT, fmt::format("Unknown benchmark \"{}\"", uriPrefix));
  }

  // Select randomly from the list of candidates.
  std::uniform_int_distribution<size_t> distribution(0, candidatesCount - 1);
  size_t index = distribution(rand_);
  return getBenchmark(candidateBenchmarks[index], benchmark);
}

std::vector<std::string> BenchmarkFactory::getBenchmarkNames() const {
  std::vector<std::string> names;
  names.reserve(unloadedBitcodePaths_.size() + benchmarks_.size());
  for (const auto& it : unloadedBitcodePaths_) {
    names.push_back(it.first);
  }
  for (const auto& it : benchmarks_) {
    names.push_back(it.first);
  }
  return names;
}

Status BenchmarkFactory::loadBenchmark(
    std::unordered_map<std::string, boost::filesystem::path>::const_iterator iterator,
    std::unique_ptr<Benchmark>* benchmark) {
  const std::string uri = iterator->first;
  const char* path = iterator->second.string().c_str();

  VLOG(2) << "loadBenchmark(" << path << ")";

  Bitcode bitcode;
  std::ifstream ifs;
  ifs.open(path);

  ifs.seekg(0, std::ios::end);
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Error reading file: \"{}\"", path));
  }

  std::streampos fileSize = ifs.tellg();
  if (!fileSize) {
    return Status(StatusCode::INVALID_ARGUMENT, fmt::format("File is empty: \"{}\"", path));
  }

  bitcode.resize(fileSize);
  ifs.seekg(0);
  ifs.read(&bitcode[0], bitcode.size());
  if (ifs.fail()) {
    return Status(StatusCode::NOT_FOUND, fmt::format("Error reading file: \"{}\"", path));
  }

  RETURN_IF_ERROR(addBitcode(uri, bitcode, fs::path(path)));

  unloadedBitcodePaths_.erase(iterator);
  *benchmark = benchmarks_.find(uri)->second.clone(workingDirectory_);
  return Status::OK;
}

size_t BenchmarkFactory::numBenchmarks() const {
  return benchmarks_.size() + unloadedBitcodePaths_.size();
}

Status BenchmarkFactory::scanSiteDataDirectory() {
  return addDirectoryOfBitcodes(kSiteBenchmarksDir);
}

}  // namespace compiler_gym::llvm_service

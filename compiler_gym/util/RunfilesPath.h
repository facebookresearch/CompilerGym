// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "boost/filesystem.hpp"

namespace compiler_gym::util {

// Resolve the path to a runfiles data path.
//
// Use environment variable COMPILER_GYM_RUNFILES=/path/to/runfiles if running
// outside of bazel.
boost::filesystem::path getRunfilesPath(const std::string& relPath);

// Resolve the path to the site data path.
//
// The site data path is used for storing persistent data files, such as
// benchmark datasets.
boost::filesystem::path getSiteDataPath(const std::string& relPath);

}  // namespace compiler_gym::util

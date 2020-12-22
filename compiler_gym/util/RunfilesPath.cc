// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "boost/filesystem.hpp"

namespace fs = boost::filesystem;

namespace compiler_gym::util {

// When running under bazel, the working directory is the root of the
// CompilerGym workspace. Back up one level so that we can reference other
// workspaces.
static const std::string kDefaultBase{"."};

fs::path getRunfilesPath(const std::string& relPath) {
  const char* base = std::getenv("COMPILER_GYM_RUNFILES");
  if (base) {
    return fs::path(base) / relPath;
  } else {
    return fs::path(relPath);
  }
}

fs::path getSiteDataPath(const std::string& relPath) {
  // NOTE(cummins): This function has a matching implementation in the Python
  // sources, compiler_gym.util.runfiles_path.get_site_data_path().
  // Any change to behavior here must be reflected in the Python version.
  const char* force = std::getenv("COMPILER_GYM_SITE_DATA");
  if (force) {
    return fs::path(force) / relPath;
  }

  const char* home = std::getenv("HOME");
  if (home) {
    return fs::path(home) / ".local/share/compiler_gym" / relPath;
  } else {
    // $HOME may not be set under testing conditions. In this case, use a
    // throwaway directory.
    return fs::temp_directory_path() / "compiler_gym" / relPath;
  }
}

}  // namespace compiler_gym::util

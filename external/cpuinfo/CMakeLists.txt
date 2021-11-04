project(gflags)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)

# Note that in the Bazel build the commit hash is
# 63b254577ed77a8004a9be6ac707f3dccc4e1fd9.
# This dependency was bumped up to 2e79955ecaec85da13ac8f1245a8b2afa10d31c2
# in order for CMake to be able to find it.
# Hopefully, this does not break the build.
ExternalProject_Add(
    cpuinfo
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}/external/cpuinfo"
    GIT_REPOSITORY "https://github.com/pytorch/cpuinfo.git"
    GIT_TAG 2e79955ecaec85da13ac8f1245a8b2afa10d31c2
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}"
)

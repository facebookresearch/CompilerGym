project(protobuf)
cmake_minimum_required(VERSION 3.15)

include(ExternalProject)
include(ProcessorCount)

ProcessorCount(_JOBS)
ExternalProject_Add(
  protobuf
  PREFIX "${CMAKE_CURRENT_BINARY_DIR}/protobuf"
  GIT_REPOSITORY "https://github.com/protocolbuffers/protobuf.git"
  GIT_TAG 7c40b2df1fdf6f414c1c18c789715a9c948a0725 #v3.19.1
  CONFIGURE_COMMAND ""
  BUILD_COMMAND ""
  INSTALL_COMMAND ""
)
# TODO(boian): To speed up reconfiguration add a timestamp and git clean hash check
# to see if this step can be skiped.
ExternalProject_Add_Step(protobuf protobuf_build
  ALWAYS FALSE
  COMMAND
    CC=\"${CMAKE_C_COMPILER}\" CXX=\"${CMAKE_CXX_COMPILER}\"
    ./autogen.sh &&
    ./configure "--prefix=${CMAKE_INSTALL_PREFIX}" &&
    make -j${_JOBS} install
  DEPENDEES update
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf"
)

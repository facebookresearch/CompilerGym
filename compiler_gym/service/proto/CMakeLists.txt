cmake_add_all_subdirs()

cmake_py_library(
    NAME "proto"
    SRCS "__init__.py"
    PUBLIC
    DEPS
      "::compiler_gym_service_py"
      "::compiler_gym_service_py_grpc"
)

proto_library(
  NAME
    compiler_gym_service
  SRCS
    compiler_gym_service.proto
  PUBLIC
)

py_proto_library(
    NAME
      compiler_gym_service_py
    DEPS
      ::compiler_gym_service
)

cc_proto_library(
  NAME
    compiler_gym_service_cc
  DEPS
    ::compiler_gym_service
  PUBLIC
)

cc_grpc_library(
    NAME compiler_gym_service_cc_grpc
    SRCS ::compiler_gym_service
    GRPC_ONLY
    PUBLIC
    DEPS ::compiler_gym_service_cc
)

py_grpc_library(
    NAME "compiler_gym_service_py_grpc"
    SRCS "::compiler_gym_service"
    DEPS "::compiler_gym_service_py"
)

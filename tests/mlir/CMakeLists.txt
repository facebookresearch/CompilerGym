if(NOT COMPILER_GYM_ENABLE_MLIR_ENV)
    return()
endif()

cg_add_all_subdirs()

cg_py_test(
  NAME
    mlir_env_test
  SRCS
    "mlir_env_test.py"
  DEPS
    compiler_gym::compiler_gym
    compiler_gym::wrappers::wrappers
    tests::pytest_plugins::common
    tests::pytest_plugins::mlir
    tests::test_main
)

cg_add_all_subdirs()

cg_py_test(
  NAME
    matmul
  SRCS
    "matmul_test.py"
  DEPS
    compiler_gym::envs::mlir::mlir
    compiler_gym::envs::mlir::datasets::datasets
    tests::pytest_plugins::common
    tests::pytest_plugins::mlir
    tests::test_main
)

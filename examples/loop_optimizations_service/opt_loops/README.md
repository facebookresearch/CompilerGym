LLVM's opt does not always enforce the unrolling or vectorization options passed as cli arguments. Hence, we created our own exeutable with custom unrolling pass in examples/loop_optimizations_service/loop_unroller that enforces the unrolling factors passed in its cli.

The tool also logs the configuration and IR of each loop to a JSON file (if you specify `--emit-json`) or YAML file (if you specify `--emit-yaml`).

## Usage

To run the custom unroller:
```
bazel run //examples/loop_optimizations_service/opt_loops:opt_loops -- <input>.ll --funroll-count=<num> --force-vector-width=<num> -S -o <output>.ll --emit-yaml=<path to output yaml file>
```

### Using the python service without bazel

1. Build the `opt_loops` custom tool that modifies and configures optimization parameters of each loop in a LLVM IR file:
Follow the [Building from source using CMake](../../INSTALL.md#building-from-source-with-cmake) instructions with `-DCOMPILER_GYM_BUILD_EXAMPLES=ON` added to the `cmake` command.

2. Run the example from the `examples` directory of the repo
```sh
$ cd examples
$ loop_optimizations_service/opt_loops -- <input>.ll --funroll-count=<num> --force-vector-width=<num> -S -o <output>.ll --emit-yaml=<path to output yaml file>
```

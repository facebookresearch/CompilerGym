# Unrolling CompilerGym Service Example

This is an example of how to create your own CompilerGym environment. All paths listed below are relative to the path of this README file.

* Actions: this environment focuses on the unrolling optimization. The actions are the different unrolling factors.
    - The actions are listed in `action_spaces` struct in `service_py/example_service.py`
    - The actions are implemented in `apply_action(...)` function in `service_py/example_service.py`
* Observations: the observations are: textual form of the LLVM IR, statistical features of different types of IR instructions, runtime execution, or code size
    - The observations are listed in `observation_spaces` struct in `service_py/example_service.py`.
    - The observations are implemented in `get_observation(...)` function in `service_py/example_service.py`
* Rewards: the rewards could be runtime or code size.
    - The rewards are implemented in `__init__.py` and they reuse the runtime and code size observations mentioned above
* Benchmarks: this environment expects your benchmarks to follow the templates from the [Neruovectorizer repo](https://github.com/intel/neuro-vectorizer/tree/master/training_data) repo, that was in turn adapted from the [LLVM loop test suite](https://github.com/llvm/llvm-test-suite/blob/main/SingleSource/UnitTests/Vectorizer/gcc-loops.cpp).
    - To implement your benchmark, you need to: include the `header.h` file, implement your benchmark in a custom function, then invoke it using `BENCH` macro inside the `main()` function.
    - Following this template is necessary in order for the benchmark to measure the execution runtime and write it to stdout, which is in turn parsed by this environment to measure the runtime reward.
    - You can view and add examples of benchmarks in `benchmarks` directory
    - Also, when adding your own benchmark, you need to add it to the `UnrollingDataset` class in `__init__.py`

## Usage

Run `example.py` example:
```sh
$ bazel run //examples/example_unrolling_service:example
```

Run `env_tests.py` unit tests:

```sh
$ bazel test //examples/example_unrolling_service:env_tests
```

### Using the python service without bazel

Because the python service contains no compiled code, it can be run directly as
a standalone script without using the bazel build system.

1. Build the `loop_unroller` custom tool that modifies the unrolling factor of each loop in a LLVM IR file:
Follow the [Building from source using CMake](../../INSTALL.md#building-from-source-with-cmake) instructions with `-DCOMPILER_GYM_BUILD_EXAMPLES=ON` added to the `cmake` command.

2. Run the example
```sh
$ cd examples
$ python3 example_compiler_gym_service/demo_without_bazel.py
```

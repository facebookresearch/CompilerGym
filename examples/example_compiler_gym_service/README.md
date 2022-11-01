# Example CompilerGym Service

CompilerGym uses a
[client/service architecture](https://facebookresearch.github.io/CompilerGym/compiler_gym/service.html).
The client is the frontend python API and the service is the backend that
handles the actual compilation. The backend exposes a
[CompilerGymService](https://github.com/facebookresearch/CompilerGym/blob/development/compiler_gym/service/proto/compiler_gym_service.proto)
RPC interface that the frontend interacts with.

This directory contains an example backend service implemented in C++ and
Python. Both implementations have the same features. They don't do any actual
compilation, but can be used as a starting point for adding support for new
compilers, or for debugging and testing frontend code.

If you have any questions please [file an
issue](https://github.com/facebookresearch/CompilerGym/issues/new/choose).


## Features

* A static action space with three items: `["a", "b", "c"]`. The action space
  never changes. Actions never end the episode.
* There are two observation spaces:
  * `ir` which returns the string "Hello, world!".
  * `features` which returns an `int64_tensor` of `[0, 0, 0]`.
* A single reward space `runtime` which returns 0.
* It has a single dataset "benchmark://example-compiler-v0" with two programs "foo" and
  "bar".


## Implementation

There are two identical service implementations, one in Python, one in C++. See
[service_cc/ExampleService.h](service_cc/ExampleService.h) for the C++ service,
and [service_py/example_service.py](service_py/example_service.py) for the
Python service. The module [__init__.py](__init__.py) defines the reward space,
dataset, and registers two new environments using these services.

The file [demo.py](demo.py) demonstrates how to use these example environments
using CompilerGym's bazel build system. The file [env_tests.py](env_tests.py)
contains unit tests for the example services. Because the Python and C++
services implement the same interface, the same tests are run against both
environments.

## Usage

Run the demo script using:

```sh
$ bazel run -c opt //examples/example_compiler_gym_service:demo
```

Run the unit tests using:

```sh
$ bazel test //examples/example_compiler_gym_service/...
```

### Using the python service without bazel

Because the python service contains no compiled code, it can be run directly as
a standalone script without using the bazel build system. From the root of the
CompilerGym repository, run:

```sh
$ cd examples
$ python3 example_compiler_gym_service/demo_without_bazel.py
```

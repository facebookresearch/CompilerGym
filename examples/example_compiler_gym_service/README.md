# Example CompilerGym Service

CompilerGym uses a
[client/service architecture](https://facebookresearch.github.io/CompilerGym/compiler_gym/service.html).
The client is the frontend python API and the service is the backend that
handles the actual compilation. The backend exposes a
[CompilerGymService](https://github.com/facebookresearch/CompilerGym/blob/development/compiler_gym/service/proto/compiler_gym_service.proto)
RPC interface that the frontend interacts with.

This directory contains an example backend service implemented in C++ and
Python. Both implementations have the same features. They don't do any actual
compilation, but can be used as a starting point for writing new services, or
for debugging and testing frontend code.

Features:

* Enforces the service contract, e.g. `StartSession()` must be called before
  `EndSession()`, list indices must be in-bounds, etc.
* Implements all of the RPC endpoints.
* It has a single dataset "benchmark://example-v0" with two programs "foo" and
  "bar".
* It has a static action space with three items: `["a", "b", "c"]`. The action
  space never changes. Actions never end the episode.
* There are two observation spaces:
    * `ir` which returns the string "Hello, world!".
    * `features` which returns an `int64_list` of `[0, 0, 0]`.
* There is a single reward space `runtime` which returns 0.
* Supports default observation and reward spaces.

See [service_cc/ExampleService.h](service_cc/ExampleService.h) for the C++
service implementation,
[service_py/example_service.py](service_py/example_service.py) for the Python
version, [__init__.py](__init__.py) for a python module that registers this
service with the gym on import, and [env_tests.py](env_tests.py) for tests.


## Usage

Start an example service using:

```sh
$ bazel run -c opt //examples/example_compiler_gym_service/service -- \
      --port=8080 --working_dir=/tmp
```

The service never terminates and does not print logging messages. Interact with
the RPC endpoints using your frontend of choice, for example, the manual
environment:

```sh
$ bazel run -c opt //compiler_gym/bin:manual_env -- --service=localhost:8080
```

Kill the service using C-c when you are done.

# Example CompilerGym Service

This directory contains an a simple example implementation of the
CompilerGymService interface, useful for debugging or testing frontend logic.

Features:

* Enforces the service contract, e.g. `Init()` must be called before
  `StartEpisode()`, list indices must be in-bounds, etc.
* It has two programs "foo" and "bar".
* It has a static action space with three items: `["a", "b", "c"]`. The action
  space never changes. Actions never end the episode.
* There are two observation spaces:
    * `ir` which returns the string "Hello, world!".
    * `features` which returns an `int64_list` of `[0, 0, 0]`.
* There is a single reward space `codesize` which returns 0.
* Supports eager observation and reward.


## Usage

Start an example service using:

```sh
$ bazel run -c opt //examples/example_compiler_gym_service -- --port=8080
```

The service does never terminates and does not print logging messages.
Interact with the RPC endpoints using your frontend of choice, for example, the
manual environment:

```sh
$ python -m compiler_gym.bin.manual_env --service=localhost:8080
```

Kill the service using C-c when you are done.

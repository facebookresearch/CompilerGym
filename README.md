![CompilerGym](https://github.com/facebookresearch/CompilerGym/raw/development/docs/source/_static/img/logo.png)

---

<!-- Documentation -->
<a href="http://facebookresearch.github.io/CompilerGym/">
    <img src="https://img.shields.io/badge/documentation-latest-blue.svg" alt="Documentation" height="20">
</a>
<!-- PyPi Version -->
<a href="https://pypi.org/project/compiler-gym/">
    <img src="https://badge.fury.io/py/compiler-gym.svg" alt="PyPI version" height="20">
</a>
<!-- CI status -->
<a href="https://github.com/facebookresearch/CompilerGym/actions?query=workflow%3ACI+branch%3Adevelopment">
    <img src="https://github.com/facebookresearch/CompilerGym/workflows/CI/badge.svg?branch=development" alt="CI status" height="20">
</a>
<!-- Downloads counter -->
<a href="https://pypi.org/project/compiler-gym/">
    <img src="https://pepy.tech/badge/compiler-gym" alt="PyPi Downloads" height="20">
</a>
<!-- license -->
<a href="https://tldrlegal.com/license/mit-license">
    <img src="https://img.shields.io/pypi/l/compiler-gym" alt="License" height="20">
</a>

CompilerGym is a toolkit for exposing compiler optimization problems
for reinforcement learning. It allows machine learning researchers to
experiment with program optimization techniques without requiring any
experience in compilers, and provides a framework for compiler
developers to expose new optimization problems for AI.


**Table of Contents**

<!-- MarkdownTOC -->

- [Getting Started](#getting-started)
  - [Installation](#installation)
    - [Building from Source](#building-from-source)
  - [Trying it out](#trying-it-out)
- [Leaderboards](#leaderboards)
  - [llvm-ic-v0](#llvm-ic-v0)
    - [cBench-v0](#cbench-v0)
- [Contributing](#contributing)
- [Citation](#citation)

<!-- /MarkdownTOC -->

# Getting Started

Starting with CompilerGym is simple. If you not already familiar with the gym
interface, refer to the
[getting started guide](http://facebookresearch.github.io/CompilerGym/getting_started.html)
for an overview of the key concepts.


## Installation

Install the latest CompilerGym release using:

    $ pip install compiler_gym

The binary works on macOS and Linux (on Ubuntu 18.04, Fedora 28, Debian 10 or
newer equivalents).

### Building from Source

If you prefer, you may build from source. This requires a modern C++ toolchain.
On macOS you can use the system compiler. On linux, install the required
toolchain using:

    $ sudo apt install clang libtinfo5 patchelf
    $ export CC=clang
    $ export CXX=clang++

We recommend using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
to manage the remaining build dependencies. First create a conda environment
with the required dependencies:

    $ conda create -n compiler_gym python=3.8 bazel=3.1.0 cmake pandoc
    $ conda activate compiler_gym

Then clone the CompilerGym source code using:

    $ git clone https://github.com/facebookresearch/CompilerGym.git
    $ cd CompilerGym

Install the python development dependencies using:

    $ make init

Then run the test suite to confirm that everything is working:

    $ make test

To build and install the python package, run:

    $ make install

**NOTE:** To use the python code that is installed by `make install` you must
leave the root directory of this repository. Attempting to import `compiler_gym`
while in the root of this repository will cause import errors.

When you are finished, you can deactivate and delete the conda
environment using:

    $ conda deactivate
    $ conda env remove -n compiler_gym


## Trying it out

In Python, import `compiler_gym` to use the environments:

```py
>>> import gym
>>> import compiler_gym                     # imports the CompilerGym environments
>>> env = gym.make("llvm-autophase-ic-v0")  # starts a new environment
>>> env.require_dataset("npb-v0")           # downloads a set of programs
>>> env.reset()                             # starts a new compilation session with a random program
>>> env.render()                            # prints the IR of the program
>>> env.step(env.action_space.sample())     # applies a random optimization, updates state/reward/actions
```

See the
[documentation website](http://facebookresearch.github.io/CompilerGym/) for
tutorials, further details, and API reference.


# Leaderboards

These leaderboards track the performance of user-submitted algorithms for
CompilerGym tasks. To submit a result please see
[this document](https://github.com/facebookresearch/CompilerGym/blob/development/CONTRIBUTING.md#leaderboard-submissions).

## llvm-ic-v0

LLVM is a popular open source compiler used widely in industry and research.
This environment exposes the optimization pipeline as a set of actions that can
be applied to a particular program. The goal of the agent is to select the
sequence of optimizations that lead to the greatest reduction in instruction
count in the program being compiled. Reward is the reduction in codesize
achieved scaled to the reduction achieved by LLVM's builtin `-Oz` pipeline.

### cBench-v0

This leaderboard tracks the results achieved by algorithms on the `llvm-ic-v0`
environment on the 23 benchmarks in the `cBench-v0` dataset.

| Author | Algorithm | Links | Mean Walltime | Geomean Improvement |
| --- | --- | --- | --- | --- |
| Facebook | Random search (t=60) | [write-up](https://github.com/facebookresearch/CompilerGym/blob/development/leaderboard/llvm_codesize/random_search/README.md), [results](https://github.com/facebookresearch/CompilerGym/blob/development/leaderboard/llvm_codesize/random_search/README.md) | **152.416s** (std: 157.540s) | **1.053x** (std: 0.120x) |
| Facebook | Random search (t=1800) | [write-up](https://github.com/facebookresearch/CompilerGym/blob/development/leaderboard/llvm_codesize/random_search/README.md), [results](https://github.com/facebookresearch/CompilerGym/blob/development/leaderboard/llvm_codesize/random_search/results_t1800_p125.csv) | TODO | TODO |


# Contributing

We welcome contributions to CompilerGym. If you are interested in contributing please see
[this document](https://github.com/facebookresearch/CompilerGym/blob/development/CONTRIBUTING.md).


# Citation

If you use CompilerGym in any of your work, please cite:

```
@Misc{CompilerGym,
  author = {Cummins, Chris and Leather, Hugh and Steiner, Benoit and He, Horace and Chintala, Soumith},
  title = {{CompilerGym}: A Reinforcement Learning Toolkit for Compilers},
  howpublished = {\url{https://github.com/facebookresearch/CompilerGym/}},
  year = {2020}
}
```

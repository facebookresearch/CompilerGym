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

- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Trying it out](#trying-it-out)
- [Leaderboards](#leaderboards)
  - [LLVM Instruction Count](#llvm-instruction-count)
- [Contributing](#contributing)
- [Citation](#citation)


# Getting Started

Starting with CompilerGym is simple. If you not already familiar with the gym
interface, refer to the
[getting started guide](http://facebookresearch.github.io/CompilerGym/getting_started.html)
for an overview of the key concepts.


## Installation

Install the latest CompilerGym release using:

    pip install -U compiler_gym

The binary works on macOS and Linux (on Ubuntu 18.04, Fedora 28, Debian 10 or
newer equivalents).

### Building from Source  <!-- omit in toc -->

If you prefer, you may build from source. This requires a modern C++ toolchain
and bazel.

#### macOS  <!-- omit in toc -->

On macOS the required dependencies can be installed using
[homebrew](https://docs.brew.sh/Installation):

```sh
brew install bazelisk zlib
export LDFLAGS="-L/usr/local/opt/zlib/lib"
export CPPFLAGS="-I/usr/local/opt/zlib/include"
export PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"
```

Now proceed to [All platforms](#all-platforms) below.

#### Linux  <!-- omit in toc -->

On debian-based linux systems, install the required toolchain using:

```sh
sudo apt install clang libtinfo5 libjpeg-dev patchelf
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O bazel
chmod +x bazel && mkdir -p ~/.local/bin && mv -v bazel ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
export CC=clang
export CXX=clang++
```

#### All platforms  <!-- omit in toc -->

We recommend using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
to manage the remaining build dependencies. First create a conda environment
with the required dependencies:

    conda create -n compiler_gym python=3.9 cmake pandoc
    conda activate compiler_gym

Then clone the CompilerGym source code using:

    git clone https://github.com/facebookresearch/CompilerGym.git
    cd CompilerGym

Install the python development dependencies using:

    make init

The `make init` target only needs to be run once on initial setup, or when
upgrading to a different CompilerGym release.

Run the test suite to confirm that everything is working:

    make test

To build and install the `compiler_gym` python package, run:

    make install

**NOTE:** To use the `compiler_gym` package that is installed by `make install`
you must leave the root directory of this repository. Attempting to import
`compiler_gym` while in the root of this repository will cause import errors.

When you are finished, you can deactivate and delete the conda
environment using:

    conda deactivate
    conda env remove -n compiler_gym


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


## LLVM Instruction Count

LLVM is a popular open source compiler used widely in industry and research. The
`llvm-ic-v0` environment exposes LLVM's optimizing passes as a set of actions
that can be applied to a particular program. The goal of the agent is to select
the sequence of optimizations that lead to the greatest reduction in instruction
count in the program being compiled. Reward is the reduction in instruction
count achieved scaled to the reduction achieved by LLVM's builtin `-Oz`
pipeline.

This leaderboard tracks the results achieved by algorithms on the `llvm-ic-v0`
environment on the 23 benchmarks in the `cBench-v1` dataset.

| Author | Algorithm | Links | Date | Walltime (mean) | Codesize Reduction (geomean) |
| --- | --- | --- | --- | --- | --- |
| Facebook | Random search (t=10800) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t10800.csv) | 2021-03 | 10,512.356s | **1.062×** |
| Facebook | Random search (t=3600) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t3600.csv) | 2021-03 | 3,630.821s | 1.061× |
| Facebook | Greedy search | [write-up](leaderboard/llvm_instcount/e_greedy/README.md), [results](leaderboard/llvm_instcount/e_greedy/results_e0.csv) | 2021-03 | 169.237s | 1.055× |
| Facebook | Random search (t=60) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t60.csv) | 2021-03 | 91.215s | 1.045× |
| Facebook | e-Greedy search (e=0.1) | [write-up](leaderboard/llvm_instcount/e_greedy/README.md), [results](leaderboard/llvm_instcount/e_greedy/results_e10.csv) | 2021-03 | 152.579s | 1.041× |
| Facebook | Random search (t=10) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t10.csv) | 2021-03 | **42.939s** | 1.031× |


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

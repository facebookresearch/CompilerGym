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
<!-- Getting started colab -->
<a href="https://colab.research.google.com/github/facebookresearch/CompilerGym/blob/stable/examples/getting-started.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20">
</a>

CompilerGym is a toolkit for exposing compiler optimization problems
for reinforcement learning. It allows machine learning researchers to
experiment with program optimization techniques without requiring any
experience in compilers, and provides a framework for compiler
developers to expose new optimization problems for AI.


**Table of Contents**

- [Features](#features)
- [Installation](#installation)
  - [Building from Source](#building-from-source)
- [Usage](#usage)
- [Leaderboards](#leaderboards)
  - [LLVM Instruction Count](#llvm-instruction-count)
- [Contributing](#contributing)
- [Citation](#citation)


# Features

With CompilerGym, building ML models for compiler research problems is as easy
as building ML models to play video games. Here are some highlights of key
features:

* **API:** uses the popular [Gym](https://gym.openai.com/) interface from OpenAI
  — use Python to write your agent.

* **Datasets:** wraps real world programs (C++ programs, TensorFlow programs,
  programs from Github, etc.) and a mainstream compiler
  ([LLVM](https://llvm.org/)), providing millions of programs for training.

* **Tasks and Actions:** interfaces the [LLVM](https://llvm.org/) compiler for
  one compiler research problem:  phase ordering (more to come). It has a large
  discrete action space.

* **Representations:** provides raw representations of programs, as well as
  multiple kinds of pre-computed features: you can focus on end-to-end deep
  learning or features + boosted trees, all the way up to graph models.

* **Rewards:** provides appropriate reward functions and loss functions out of
  the box.

* **Testing:** provides a validation process for correctness of results.

* **Baselines:** provides some baselines and reports their performance.

* **Competition:** provides [leaderboards](#leaderboards) for you to submit your
  results.

For a glimpse of what's to come, check out [our
roadmap](https://github.com/facebookresearch/CompilerGym/projects/1).

# Installation

Install the latest CompilerGym release using:

    pip install -U compiler_gym

The binary works on macOS and Linux (on Ubuntu 18.04, Fedora 28, Debian 10 or
newer equivalents).

## Building from Source

If you prefer, you may build from source. This requires a modern C++ toolchain
and bazel.

### macOS  <!-- omit in toc -->

On macOS the required dependencies can be installed using
[homebrew](https://docs.brew.sh/Installation):

```sh
brew install bazelisk zlib
export LDFLAGS="-L/usr/local/opt/zlib/lib"
export CPPFLAGS="-I/usr/local/opt/zlib/include"
export PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"
```

Now proceed to [All platforms](#all-platforms) below.

### Linux  <!-- omit in toc -->

On debian-based linux systems, install the required toolchain using:

```sh
sudo apt install clang-9 libtinfo5 libjpeg-dev patchelf
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O bazel
chmod +x bazel && mkdir -p ~/.local/bin && mv -v bazel ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"
export CC=clang
export CXX=clang++
```

### All platforms  <!-- omit in toc -->

We recommend using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
to manage the remaining build dependencies. First create a conda environment
with the required dependencies:

    conda create -n compiler_gym python=3.9 cmake pandoc
    conda activate compiler_gym

Then clone the CompilerGym source code using:

    git clone https://github.com/facebookresearch/CompilerGym.git
    cd CompilerGym

There are two primary git branches: `stable` tracks the latest release;
`development` is for bleeding edge features that may not yet be mature. Checkout
your preferred branch and install the python development dependencies using:

    git checkout stable
    make init

The `make init` target only needs to be run once on initial setup, or when
pulling remote changes to the CompilerGym repository.

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

# Usage

Starting with CompilerGym is simple. If you not already familiar with the gym
interface, refer to the [getting started
guide](http://facebookresearch.github.io/CompilerGym/getting_started.html) for
an overview of the key concepts.

In Python, import `compiler_gym` to use the environments:

```py
>>> import gym
>>> import compiler_gym                      # imports the CompilerGym environments
>>> env = gym.make(                          # creates a new environment
...     "llvm-v0",                           # selects the compiler to use
...     benchmark="cbench-v1/qsort",         # selects the program to compile
...     observation_space="Autophase",       # selects the observation space
...     reward_space="IrInstructionCountOz", # selects the optimization target
... )
>>> env.reset()                              # starts a new compilation session
>>> env.render()                             # prints the IR of the program
>>> env.step(env.action_space.sample())      # applies a random optimization, updates state/reward/actions
```

See the [documentation website](http://facebookresearch.github.io/CompilerGym/)
for tutorials, further details, and API reference. Our
[roadmap](https://facebookresearch.github.io/CompilerGym/about.html#roadmap) of
planned features is public, and the
[changelog](https://github.com/facebookresearch/CompilerGym/blob/development/CHANGELOG.md)
summarizes shipped features.


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
environment on the 23 benchmarks in the `cbench-v1` dataset.

| Author | Algorithm | Links | Date | Walltime (mean) | Codesize Reduction (geomean) |
| --- | --- | --- | --- | --- | --- |
| Facebook | Random search (t=10800) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t10800.csv) | 2021-03 | 10,512.356s | **1.062×** |
| Facebook | Random search (t=3600) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t3600.csv) | 2021-03 | 3,630.821s | 1.061× |
| Facebook | Greedy search | [write-up](leaderboard/llvm_instcount/e_greedy/README.md), [results](leaderboard/llvm_instcount/e_greedy/results_e0.csv) | 2021-03 | 169.237s | 1.055× |
| Facebook | Random search (t=60) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t60.csv) | 2021-03 | 91.215s | 1.045× |
| Facebook | e-Greedy search (e=0.1) | [write-up](leaderboard/llvm_instcount/e_greedy/README.md), [results](leaderboard/llvm_instcount/e_greedy/results_e10.csv) | 2021-03 | 152.579s | 1.041× |
| Jiadong Guo | Tabular Q (N=5000, H=10) | [write-up](leaderboard/llvm_instcount/tabular_q/README.md), [results](leaderboard/llvm_instcount/tabular_q/results-H10-N5000.csv) | 2021-04 | 2534.305 | 1.036× |
| Facebook | Random search (t=10) | [write-up](leaderboard/llvm_instcount/random_search/README.md), [results](leaderboard/llvm_instcount/random_search/results_p125_t10.csv) | 2021-03 | **42.939s** | 1.031× |
| Jiadong Guo | Tabular Q (N=2000, H=5) | [write-up](leaderboard/llvm_instcount/tabular_q/README.md), [results](leaderboard/llvm_instcount/tabular_q/results-H5-N2000.csv) | 2021-04 | 694.105 | 0.988× |


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

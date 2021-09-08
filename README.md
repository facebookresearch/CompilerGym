![CompilerGym](https://github.com/facebookresearch/CompilerGym/raw/development/docs/source/_static/img/logo.png)

<p align="center">
  <!-- PyPi Version -->
  <a href="https://pypi.org/project/compiler-gym/">
      <img src="https://badge.fury.io/py/compiler-gym.svg" alt="PyPI version" height="20">
  </a>
  <!-- Downloads counter -->
  <a href="https://pypi.org/project/compiler-gym/">
      <img src="https://pepy.tech/badge/compiler-gym" alt="PyPi Downloads" height="20">
  </a>
  <!-- license -->
  <a href="https://tldrlegal.com/license/mit-license">
      <img src="https://img.shields.io/pypi/l/compiler-gym" alt="License" height="20">
  </a>
  <!-- CI status -->
  <a href="https://github.com/facebookresearch/CompilerGym/actions?query=workflow%3ACI+branch%3Adevelopment">
      <img src="https://github.com/facebookresearch/CompilerGym/workflows/CI/badge.svg?branch=development" alt="CI status" height="20">
  </a>
  <!-- Getting started colab -->
  <a href="https://colab.research.google.com/github/facebookresearch/CompilerGym/blob/stable/examples/getting-started.ipynb">
      <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20">
  </a>
</p>

<p align="center">
  <i>Reinforcement learning environments for compiler optimization tasks.</i>
</p>
<p align="center">
  <i>
    Check
    <a href="http://facebookresearch.github.io/CompilerGym/">the website</a>
    for more information.
  </i>
</p>


## Introduction

CompilerGym is a library of easy to use and performant reinforcement learning
environments for compiler tasks. It allows ML researchers to interact with
important compiler optimization problems in a language and vocabulary with which
they are comfortable, and provides a toolkit for systems developers to expose
new compiler tasks for ML research. We aim to act as a catalyst for making
compilers faster using ML. Key features include:

* **Ease of use:** built on the the popular [Gym](https://gym.openai.com/)
  interface - use Python to write your agent. With CompilerGym, building ML
  models for compiler research problems is as easy as building ML models to play
  video games.

* **Batteries included:** includes everything required to get started. Wraps
  real world programs and compilers to provide millions of instances for
  training. Provides multiple kinds of pre-computed program representations: you
  can focus on end-to-end deep learning or features + boosted trees, all the way
  up to graph models. Appropriate reward functions and loss functions for
  optimization targets are provided out of the box.

* **Reproducible:** provides validation for correctness of results, common
  baselines, and [leaderboards](#leaderboards) for you to submit your results.

For a glimpse of what's to come, check out [our
roadmap](https://github.com/facebookresearch/CompilerGym/projects/1).


## Installation

Install the latest CompilerGym release using:

    pip install -U compiler_gym

See
[INSTALL.md](https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md)
for further details.


## Usage

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
for tutorials, further details, and API reference. See the [examples](/examples)
directory for pytorch integration, agent implementations, etc.


## Leaderboards

These leaderboards track the performance of user-submitted algorithms for
CompilerGym tasks. To submit a result please see
[this document](https://github.com/facebookresearch/CompilerGym/blob/development/CONTRIBUTING.md#leaderboard-submissions).


### LLVM Instruction Count

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
| Patrick Hesse | DQN (N=4000, H=10) | [write-up](leaderboard/llvm_instcount/dqn/README.md), [results](leaderboard/llvm_instcount/dqn/results-instcountnorm-H10-N4000.csv) | 2021-06 | 91.018s | 1.029× |
| Jiadong Guo | Tabular Q (N=2000, H=5) | [write-up](leaderboard/llvm_instcount/tabular_q/README.md), [results](leaderboard/llvm_instcount/tabular_q/results-H5-N2000.csv) | 2021-04 | 694.105 | 0.988× |


## Contributing

We welcome contributions to CompilerGym. If you are interested in contributing please see
[this document](https://github.com/facebookresearch/CompilerGym/blob/development/CONTRIBUTING.md).


## Citation

If you use CompilerGym in any of your work, please cite:

```
@Misc{CompilerGym,
  author = {Cummins, Chris and Leather, Hugh and Steiner, Benoit and He, Horace and Chintala, Soumith},
  title = {{CompilerGym}: A Reinforcement Learning Toolkit for Compilers},
  howpublished = {\url{https://github.com/facebookresearch/CompilerGym/}},
  year = {2020}
}
```

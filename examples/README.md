# CompilerGym Examples <!-- omit in toc -->

This directory contains code samples for everything from implementing simple
RL agents to adding support for entirely new compilers. Is there an example that
you think is missing? If so, please [contribute](/CONTRIBUTING.md)!


**Table of contents:**

- [Autotuning](#autotuning)
  - [Performing a random walk of an environment](#performing-a-random-walk-of-an-environment)
  - [GCC Autotuning (genetic algorithms, hill climbing, + more)](#gcc-autotuning-genetic-algorithms-hill-climbing--more)
  - [Makefile integration](#makefile-integration)
  - [Random search using the LLVM C++ API](#random-search-using-the-llvm-c-api)
- [Reinforcement learning](#reinforcement-learning)
  - [PPO and integration with RLlib](#ppo-and-integration-with-rllib)
  - [Actor-critic](#actor-critic)
  - [Tabular Q learning](#tabular-q-learning)
- [Extending CompilerGym](#extending-compilergym)
  - [Example CompilerGym service](#example-compilergym-service)
  - [Example loop unrolling](#example-loop-unrolling)
- [Miscellaneous](#miscellaneous)
  - [Exhaustive search of bounded action spaces](#exhaustive-search-of-bounded-action-spaces)
  - [Estimating the immediate and cumulative reward of actions and benchmarks](#estimating-the-immediate-and-cumulative-reward-of-actions-and-benchmarks)


## Autotuning

### Performing a random walk of an environment

The [random_walk.py](random_walk.py) script runs a single episode of a
CompilerGym environment, logging the action taken and reward received at each
step. Example usage:

```sh
$ python random_walk.py --env=llvm-v0 --step_min=100 --step_max=100 \
      --benchmark=cbench-v1/dijkstra --reward=IrInstructionCount

=== Step 1 ===
Action:       -lower-constant-intrinsics (changed=False)
Reward:       0.0
Step time:    805.6us

=== Step 2 ===
Action:       -forceattrs (changed=False)
Reward:       0.0
Step time:    229.8us

...

=== Step 100 ===
Action:       -globaldce (changed=False)
Reward:       0.0
Step time:    213.9us

Completed 100 steps in 91.6ms (1091.3 steps / sec).
Total reward: 161.0
Max reward:   111.0 (+68.94% at step 31)
```

For further details run: `python random_walk.py --help`.


### GCC Autotuning (genetic algorithms, hill climbing, + more)

The [gcc_search.py](gcc_search.py) script contains implementations of several
autotuning techniques for the GCC environment. It was used to produce the
results for the GCC experiments in the [CompilerGym
whitepaper](https://arxiv.org/pdf/2109.08267.pdf). For further details run:
`python gcc_search.py --help`.


### Makefile integration

The [makefile_integration](makefile_integration/) directory demonstrates a
simple integration of CopmilerGym into a C++ Makefile config. For details see
the [Makefile](makefile_integration/Makefile).


### Random search using the LLVM C++ API

While not intended for the majority of users, it is entirely straightforward to
skip CompilerGym's Python frontend and interact with the C++ APIs directly. The
[RandomSearch.cc](RandomSearch.cc) file demonstrates a simple parallelized
random search implemented for the LLVM compiler service. Run it using:

```
bazel run -c opt //examples:RandomSearch -- --benchmark=benchmark://cbench-v1/crc32
```

For further details run: `bazel run -c opt //examples:RandomSearch -- --help`


## Reinforcement learning


### PPO and integration with RLlib

<a href="https://colab.research.google.com/github/facebookresearch/CompilerGym/blob/stable/examples/getting-started.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Colab" height="20">
</a>

The [rllib.ipynb](rllib.ipynb) notebook demonstrates integrating CompilerGym
with the popular [RLlib](https://docs.ray.io/en/master/rllib.html) reinforcement
learning library. In notebook covers registering a custom environment using a
constrained subset of the LLVM environment's action space a finite time horizon,
and trains a PPO agent using separate train/val/test datasets.


### Actor-critic

The [actor_critic](actor_critic.py) script contains a simple actor-critic
example using PyTorch. The objective is to minimize the size of a benchmark
(program) using LLVM compiler passes. At each step there is a choice of which
pass to pick next and an episode consists of a sequence of such choices,
yielding the number of saved instructions as the overall reward. For
simplification of the learning task, only a (configurable) subset of LLVM passes
are considered and every episode has the same (configurable) length.

For further details run: `python actor_critic.py --help`.


### Tabular Q learning

The [tabular_q](tabular_q.py) script contains a simple tabular Q learning
example for the LLVM environment. Using selected features from Autophase
observation space, given a specific training program as gym environment, find
the best action sequence using online Q learning.

For further details run: `python tabular_q.py --help`.


## Extending CompilerGym


### Example CompilerGym service

The [example_compiler_gym_service](example_compiler_gym_service) directory
demonstrates how to extend CompilerGym with support for new compiler problems.
The directory contains bare bones implementations of backends in Python or C++
that can be used as the basis for adding new compiler environments. See the
[README.md](example_compiler_gym_service/README.md) file for further details.


### Example loop unrolling

The [example_unrolling_service](example_unrolling_service) directory
demonstrates how to implement support for a real compiler problem by integrating
with commandline loop unrolling flags for the LLVM compiler. See the
[README.md](example_unrolling_service/README.md) file for further details.


## Miscellaneous


### Exhaustive search of bounded action spaces

The [brute_force.py](brute_force.py) script runs a parallelized brute force of
an action space. It enumerates all possible combinations of actions up to a
finite episode length and evaluates them, logging the incremental rewards of
each. Example usage:

```
$ python brute_force.py --env=llvm-ic-v0 --benchmark=cbench-v1/dijkstra \
      --episode_length=8 --brute_force_action_list=-sroa,-mem2reg,-newgvn

Enumerating all episodes of 3 actions x 8 steps
Started 24 brute force workers for benchmark benchmark://cbench-v1/dijkstra using reward IrInstructionCountOz.
=== Running 6,561 trials ===
Runtime: 8 seconds. Progress: 100.00%. Best reward found: 0.8571428571428572.
Ending jobs ... I1014 12:04:51.671775 3245811 CreateAndRunCompilerGymServiceImpl.h:128] Service "/dev/shm/compiler_gym_cec/s/1014T120451-646797-5770" listening on 37505, PID = 3245811
completed 6,561 of 6,561 trials (100.000%), best sequence -mem2reg -mem2reg -sroa -sroa -mem2reg -sroa -sroa -newgvn
```

For further details run: `python brute_force.py --help`.

The [explore.py](explore.py) script evaluates all possible combinations of
actions up to a finite limit, but partial sequences of actions that end up in
the same state are deduplicated, sometimes dramatically reducing the size of the
search space. This script can also be configured to do a beam search.

Example usage:

```
$ python explore.py --env=llvm-ic-v0 --benchmark=cbench-v1/dijkstra \
      --episode_length=8 --explore_actions=-simplifycfg,-instcombine,-mem2reg,-newgvn

...

*** Processing depth 6 of 8 with 11 states and 4 actions.

                                 unpruned     self_pruned    cross_pruned     back_pruned         dropped             sum
        added this depth                0              33               0              11               0              44
   full nodes this depth                0           2,833           1,064             199               0           4,096
     added across depths               69             151              23              34               0             277
full added across depths               69           3,727           1,411             254               0           5,461

Time taken for depth: 0.05 s
Top 3 sequence(s):
  0.9694  -mem2reg, -newgvn, -simplifycfg, -instcombine
  0.9694  -newgvn, -instcombine, -mem2reg, -simplifycfg
  0.9694  -newgvn, -instcombine, -mem2reg, -simplifycfg, -instcombine


*** Processing depth 7 of 8 with 0 states and 4 actions.

There are no more states to process, stopping early.
```

For further details run: `python brute_force.py --help`.


### Estimating the immediate and cumulative reward of actions and benchmarks

The [sensitivity_analysis](sensitivity_analysis/) directory contains a pair of
scripts for estimating the sensitivity of the reward signal to different
environment parameters:

* [action_sensitivity_analysis.py](sensitivity_analysis/action_sensitivity_analysis.py):
  This script estimates the immediate reward that running a specific action has
  by running trials. A trial is a random episode that ends with the determined
  action.
* [benchmark_sensitivity_analysis.py](sensitivity_analysis/benchmark_sensitivity_analysis.py):
  This script estimates the cumulative reward for a random episode on a
  benchmark by running trials. A trial is an episode in which a random number of
  random actions are performed and the total cumulative reward is recorded.

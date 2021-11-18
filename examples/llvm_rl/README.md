# Reinforcement Learning experiments using RLlib  <!-- omit in toc -->

This directory contains code for running reinforcement learning experiments on
CompilerGym using RLlib.

- [About](#about)
- [Usage](#usage)
  - [Training for 1k episodes using the default on the local machine](#training-for-1k-episodes-using-the-default-on-the-local-machine)
  - [Sweeping the learning rate of an algorithm](#sweeping-the-learning-rate-of-an-algorithm)
  - [Re-running a previous configuration](#re-running-a-previous-configuration)
  - [Testing](#testing)
  - [Debugging your code/config](#debugging-your-codeconfig)
- [Training Directory Layout](#training-directory-layout)
- [Reproducing the experiments from our paper](#reproducing-the-experiments-from-our-paper)
  - [Section VII.G: Effect of algorithm](#section-viig-effect-of-algorithm)
  - [Section VII.H: Effect of training set](#section-viih-effect-of-training-set)
  - [Section VII.I: Effect of representation space](#section-viii-effect-of-representation-space)


## About

This code is 95% config parsing and wrangling, with the final 5% being the
`tune.run()` invocation to do the actual learning.

The configuration of experiments is decomposed into five subunits:

 * **Executor**: controls where work gets done (e.g. submitting jobs to a slurm
   cluster vs running on local machine).
 * **Environment**: describes the gym environment that is interacted with.
 * **Agent**: specifies the RLlib agent and its configuration.
 * **Training**: describes the training loop, i.e. how long to train for, and
   over what benchmarks.
 * **Testing**: describes the final testing configuration.

Each of the five subunits is defined by a
[pydantic](https://pydantic-docs.helpmanual.io/) model which describes the
schema of the config subunit. E.g. [model/environment.py](model/environment.py)
defines the model describing an environment. Each of the "fields" which make up
a config subunit is documented and comes with runtime validation of inputs, so
reading through the files in the models directory is the best way to understand
what is going on.

[Hydra](https://github.com/facebookresearch/hydra) is used to generate the
inputs to these models from YAML files. The `config` directory contains a set of
YAML files for common subunit configs which can be composed or overriden from
the commandline. E.g. [config/testing/cbench.yaml](config/testing/cbench.yaml)
contains the configuration for a `Testing` model that uses the cBench
benchmarks.

Once all of this is combined, running the actual RL jobs is pretty
straightforward. Configuration can be done by modifying the YAML config files
and overriding the
[defaults](https://hydra.cc/docs/tutorials/basic/your_first_app/defaults), or by
specifying individual options at the commandline.


## Usage

There are three scripts: [train.py](train.py) which contains the training
script, [test.py](test.py) for testing trained agents, and [info.py](info.py)
for summarizing results.

Here are some example command line invocations to get started.


### Training for 1k episodes using the default on the local machine

```sh
python -m llvm_rl.train -m training.episodes=1000
```


### Sweeping the learning rate of an algorithm

The
[multi-run](https://hydra.cc/docs/tutorials/basic/running_your_app/multi-run)
feature can be used to create parameter sweeps. For example, train three agents
with different learning rates using:

```sh
python -m llvm_rl.train -m agent.type=ppo agent.args.lr=0.001,0.0005,0.0001
```


### Re-running a previous configuration

To re-run a previous experiment configuration, navigate to the generated hydra
subdirectory and set that as the `--config-path`:

```sh
python -m llvm_rl.train \
    --config-path /path/to/previous/experiment/run/config-0/hydra/ \
    -m
```

You can still override individual configuration options as normal. E.g. to
re-run a previous job but using a different number of training benchmarks:

```sh
python -m llvm_rl.train \
    --config-path /path/to/previous/experiment/run/config-0/hydra/ \
    -m training.episodes=5000
```


### Testing

Use the [test.py](test.py) script to test a trained agent:

```sh
python -m llvm_rl.test /path/to/previous/experiment/run/
```

Test results are cached, so that work is only done on the first run for each
experiment.

If run with no arguments, every trained agent in the default logging directory
(`~/logs/compiler_gym/llvm_rl`) will be tested:

```sh
python -m llvm_rl.test
```


### Debugging your code/config

A "no-op" executor can be used for debugging and validating your config files
without doing any expensive work (it simply discards the jobs that are submitted
to it):

```sh
python -m llvm_rl.train -m executor.type=noop
```


## Training Directory Layout

The `outputs` config option sets the base output directory (default:
`~/logs/compiler_gym/llvm_rl`) and `experiment` is an arbitrary string used for
grouping results (if not specified, defaults to `unnamed_experiment`). Jobs
generate files using this layout:

```sh
# The top level outputs for this run:
${outputs}/${experiment}/${YYY-MM-DD}/${hh-mm-ss}/
# Logs generated by the slurm worker:
${outputs}/${experiment}/${YYY-MM-DD}/${hh-mm-ss}/slurm_logs
# Each configuration of a multi-run gets its own directory using an ascending job ID:
${outputs}/${experiment}/${YYY-MM-DD}/${hh-mm-ss}/config-${job_id}
# A copy of all logging messages from the training job:
${outputs}/${experiment}/${YYY-MM-DD}/${hh-mm-ss}/config-${job_id}/train.log
# This directory contains an exact copy of the configs used:
${outputs}/${experiment}/${YYY-MM-DD}/${hh-mm-ss}/config-${job_id}/hydra
# RLlib checkpoints:
${outputs}/${experiment}/${YYY-MM-DD}/${hh-mm-ss}/config-${job_id}/train/${experiment}-0_0_2021-07-23_18-11-04
```


## Reproducing the experiments from our paper

This section describes how to reproduce the experiments in [our
paper](https://arxiv.org/pdf/2109.08267.pdf):


### Section VII.G: Effect of algorithm

```sh
python -m llvm_rl.train -m \
    experiment=algo \
    agent=a2c,apex,impala,ppo \
    num_replicas=10 \
    training=csmith \
    testing=all
```

### Section VII.H: Effect of training set

```sh
python -m llvm_rl.train -m \
    experiment=training_set \
    agent=ppo \
    num_replicas=10 \
    training=csmith,github,tensorflow \
    testing=csmith,github,tensorflow
```


### Section VII.I: Effect of representation space

```sh
python -m llvm_rl.train -m \
    experiment=observation_space \
    agent=ppo \
    num_replicas=10 \
    environment=autophase,autophase-with-history,instcount,instcount-with-history \
    training=csmith \
    testing=cbench
```

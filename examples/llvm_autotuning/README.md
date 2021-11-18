# Autotuning for LLVM Phase Ordering <!-- omit in toc -->

This directory contains various autotuners for the LLVM phase ordering
environment by integrating with third-party libraries or implementing simple
search strategies.

- [Autotuners](#autotuners)
- [Installation](#installation)
- [Usage](#usage)
  - [Running autotuning experiments](#running-autotuning-experiments)
  - [Overview of results](#overview-of-results)
- [Reproducing the experiments from our paper](#reproducing-the-experiments-from-our-paper)


## Autotuners

The following autotuning techniques are supported:

1. Greedy Search: At each step evaluate all possible actions and select the
   action which provides the greatest reward, terminating once no positive
   reward can be achieved by any action.
1. [Nevergrad](https://facebookresearch.github.io/nevergrad/): A gradient-free
   optimization library that supports many optimization strategies.
1. [OpenTuner](https://opentuner.org/): An extensible framework for program
   autotuning.
1. Random Search: randomly select actions randomly until a configurable number
   of steps have elapsed without a positive reward.


## Installation

Using these scripts requires installing the package of examples code. From the
root of this repository:

```
cd examples
python setup.py install
```


## Usage

There are two scripts, [tune.py](tune.py) and [info.py](info.py). The tuning
script is used to run the autotuning experiments and record log the results to
CSV files. The info script reads and aggregates results over many multiple CSV
files.

### Running autotuning experiments

Invoke the tuning script using:

```
python -m llvm_autotuning.tune -m <arguments>
```

Where `<arguments>` is a list of configuration arguments. Key configuration
options are:

| Argument                        | Description                                                                 |
|---------------------------------|-----------------------------------------------------------------------------|
| `autotuner`                     | Name of the autotuner. One of: {`greedy`,`nevergrad`,`opentuner`,`random`}. |
| `autotuner.optimization_target` | Loss function to optimize for. One of: {`codesize`,`binsize`,`runtime`}.    |
| `autotuner.search_time_seconds` | Number of seconds to run on each program instance. Default: 3600.           |
| `outputs`                       | Base output directory. Default: `~/logs/compiler_gym/llvm_autotuning`.      |
| `experiment`                    | Name of the experiment, used to determine output directory.                 |
| `num_replicas`                  | Number of times to repeat each experiment. Default: 10.                     |
| `executor.cpus`                 | Number of parallel experiment workers. Default: #. cores on machine.        |

For example, to run 10 minutes of code size autotuning using
[Nevergrad](https://facebookresearch.github.io/nevergrad/) using 32 parallel
worker processes, logging results to /tmp/logs/my-experiment:

```
python -m llvm_autotuning.tune -m \
    experiment=my-experiment \
    outputs=/tmp/logs \
    executor.cpus=32 \
    num_replicas=1 \
    autotuner=nevergrad \
    autotuner.optimization_target=codesize \
    autotuner.search_time_seconds=600
```

Multiple values can be passed to each configuration option, defining a grid of
unique configurations that will each be run. For example, the following
configuration creates a grid sweep over the search time (5 minutes and 10
minutes) and episode length (100 steps, 200 steps, and 300 steps):

```
python -m llvm_autotuning.tune -m \
    experiment=my-tuning-experiment \
    num_replicas=10 \
    benchmarks=csmith-50 \
    autotuner=nevergrad \
    autotuner.optimization_target=codesize \
    autotuner.search_time_seconds=300,600 \
    autotuner.algorithm_config.episode_length=100,200,300
```

Use `--help` to see the full list of configuration options.


### Overview of results

Summarize autotuning results using:

```
python -m llvm_autotuning.info
```

This aggregates results in the default `~/logs/compiler_gym/llvm_autotuning`
directory. Specify one or more different directories as command line arguments,
e.g.:

```
python -m llvm_autotuning.info /path/to/logs/dir/a ~/logs_dir_b
```


## Reproducing the experiments from our paper

To reproduce the experiments in Section VII.C of [our
paper](https://arxiv.org/pdf/2109.08267.pdf), run:

```sh
export AUTOTUNER=greedy; export TARGET=codesize; \
python -m llvm_autotuning.tune -m \
    experiment="$AUTOTUNER-${TARGET}" \
    autotuner="$AUTOTUNER" \
    autotuner.optimization_target="$TARGET"
```

where `AUTOTUNER` is one of `greedy`, `random`, `nevergrad`, or `opentuner`; and
`TARGET` is one of `codesize`, `binsize`, or `runtime`. The parameters for each
of the autotuners defaults to those identified by sweeping the hyperparameters
on a holdout set of 50 CSmith benchmarks using codesize reward.

To reproduce the hyperparameter sweeps for random search:

```sh
export TARGET=codesize; \
python -m llvm_autotuning.tune -m \
    experiment=random-"${TARGET}"-tuning \
    autotuner=random \
    autotuner.optimization_target="$TARGET" \
    benchmarks=csmith-50 \
    autotuner.algorithm_config.patience=5,10,25,50,75,100,125,150,175,200,225,250,275,300,325,350
```

To reproduce the hyperparameter sweeps for
[Nevergrad](https://facebookresearch.github.io/nevergrad/):

```sh
export TARGET=codesize; \
python -m llvm_autotuning.tune -m \
    experiment=nevergrad-"${TARGET}"-tuning \
    autotuner=nevergrad \
    autotuner.optimization_target="$TARGET" \
    benchmarks=csmith-50 \
    autotuner.algorithm_config.episode_length=50,100,150,200,250,300,350,400 \
    autotuner.algorithm_config.optimizer=DiscreteOnePlusOne,PortfolioDiscreteOnePlusOne,DiscreteLenglerOnePlusOne,AdaptiveDiscreteOnePlusOne,AnisotropicAdaptiveDiscreteOnePlusOne,DiscreteBSOOnePlusOne,DiscreteDoerrOnePlusOne,OptimisticDiscreteOnePlusOne,NoisyDiscreteOnePlusOne,DoubleFastGADiscreteOnePlusOne,RecombiningPortfolioOptimisticNoisyDiscreteOnePlusOne,MultiDiscrete,PSO,DE,NGOpt,TwoPointsDE,CMandAS2
```

To reproduce the hyperparameter sweep for [OpenTuner](https://opentuner.org/):

```sh
export TARGET=codesize; \
python -m llvm_autotuning.tune -m \
    experiment=opentuner-"${TARGET}"-tuning \
    autotuner=opentuner \
    autotuner.optimization_target="$TARGET" \
    benchmarks=csmith-50 \
    autotuner.algorithm_config.max_copies_of_pass=1,2,4,8,16,32,64
```

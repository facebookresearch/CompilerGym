# Autotuning for GCC command line flags <!-- omit in toc -->

This directory contains scripts to autotune the GCC environment using black box
optimization strategies.

- [Usage](#usage)
  - [Running autotuning experiments](#running-autotuning-experiments)
  - [Overview of results](#overview-of-results)
- [Reproducing the experiments from our paper](#reproducing-the-experiments-from-our-paper)


## Usage

There are two scripts, [tune.py](tune.py) and [info.py](info.py). The tuning
script is used to run the autotuning experiments and record log the results to
CSV files. The info script reads and aggregates results over many multiple CSV
files.


### Running autotuning experiments

Invoke the tuning script using:

```
python -m gcc_autotuning.tune <arguments>
```

Where `<arguments>` is a list of configuration arguments. Key configuration
options are:

| Argument                  | Description                                                            |
|---------------------------|------------------------------------------------------------------------|
| `--gcc_bin`               | Binary to use for GCC. Use `docker:<image>` for docker.                |
| `--gcc_benchmark`         | List of benchmarks to autotune. Defaults to the 12 CHStone benchmarks. |
| `--search`                | Types of autotuning to perform.                                        |
| `--gcc_search_budget`     | Maximum number of compilations per benchmark.                          |
| `--gcc_search_reptitions` | Number of times to repeat each search.                                 |
| `--objective`             | Objective function to optimize for. One of {obj_size,asm_size}.        |
| `--output_dir`            | Base output directory. Default: `~/logs/compiler_gym/gcc_autotuning`.  |

For example, to run a genetic algorithm search with a budget of 100 compilations
on two CHStone benchmarks:

```sh
python -m gcc_autotuning.tune \
    --gcc_bin=docker:gcc:11.2.0 \
    --seed=204 \
    --search=genetic \
    --gcc_search_budget=100 \
    --gcc_search_repetitions=3 \
    --gcc_benchmark=benchmark://chstone-v0/aes,benchmark://chstone-v0/sha
```

See `--helpfull` for a full list of arguments.


### Overview of results

Summarize autotuning results using:

```
python -m gcc_autotuning.info
```

This aggregates results in the default `~/logs/compiler_gym/gcc_autotuning`
directory. Specify one or more different directories as command line arguments,
e.g.:

```
python -m gcc_autotuning.info /path/to/logs/dir/a ~/logs_dir_b
```


## Reproducing the experiments from our paper

To reproduce the experiments in Section VII.D of [our
paper](https://arxiv.org/pdf/2109.08267.pdf), run:

```sh
python -m gcc_autotuning.tune \
    --gcc_bin=docker:gcc:11.2.0 \
    --seed=204 \
    --search=random,hillclimb,genetic \
    --gcc_search_budget=1000 \
    --gcc_search_repetitions=3 \
    --gcc_benchmark=benchmark://chstone-v0/adpcm,benchmark://chstone-v0/aes,benchmark://chstone-v0/blowfish,benchmark://chstone-v0/dfadd,benchmark://chstone-v0/dfdiv,benchmark://chstone-v0/dfmul,benchmark://chstone-v0/dfsin,benchmark://chstone-v0/gsm,benchmark://chstone-v0/jpeg,benchmark://chstone-v0/mips,benchmark://chstone-v0/motion,benchmark://chstone-v0/sha
```

Then use the [info.py](info.py) script to print the results and compare against
Table V in the paper:

```sh
python -m gcc_autouning.info
```

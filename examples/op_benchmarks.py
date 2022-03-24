# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This script runs microbenchmarks of CompilerGym environment operations.

To collect new measurements, run one of the following commands:

    $ python -m op_benchmarks {run,init,reset,step,observations} --env=llvm-v0 --n=100

To aggregate results from prior runs:

    $ python -m op_benchmarks info
"""
import logging
import os
import re
from collections import defaultdict
from itertools import islice
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import typer
from tabulate import tabulate

import compiler_gym
from compiler_gym import CompilerEnv
from compiler_gym.datasets import BenchmarkInitError
from compiler_gym.util.executor import Executor
from compiler_gym.util.logging import init_logging
from compiler_gym.util.runfiles_path import create_user_logs_dir
from compiler_gym.util.timer import Timer

app = typer.Typer()

logger = logging.getLogger(__name__)


def get_runtimes(op: Callable[[], Any], n: int):
    """Run `n` reptitions of function `op`, ignoring any errors."""
    runtimes = []
    for _ in range(n):
        try:
            with Timer() as timer:
                op()
            runtimes.append(timer.time)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Op failed: %s", e)
    return runtimes


@app.command()
def init(
    n: int = int(1e6),
    j: int = cpu_count(),
    env: str = "llvm-autophase-ic-v0",
    outdir: Optional[Path] = None,
):
    """Benchmark the environment startup time."""
    executor = Executor(type="local", cpus=j)
    outdir = Path(outdir or create_user_logs_dir("op_benchmarks"))
    with executor.get_executor(logs_dir=outdir) as session:
        _init(n=n, outdir=outdir, j=j, env_name=env, session=session)


def _init(n: int, outdir: Path, j: int, env_name: str, session: Executor):
    outdir.mkdir(exist_ok=True, parents=True)
    for i in range(1, j + 1):
        session.submit(
            _init_worker,
            env_name=env_name,
            n=int(ceil(n / j)),
            outfile=outdir / f".op:1:startup-shard-{i:02d}.txt",
        )


def _init_worker(env_name: str, n: int, outfile: Path):
    with open(outfile, "w") as f:
        for _ in range(0, n, min(100, n)):
            runtimes = get_runtimes(
                lambda: compiler_gym.make(env_name).close(), min(100, n)
            )
            print("\n".join(f"{x:.8f}" for x in runtimes), file=f, flush=True)


def get_benchmarks(env_name: str, n: int, seed: int, outdir: Path) -> List[str]:
    """Get `n` benchmarks from all datasets.

    If the dataset is smaller than `n`, benchmarks are repeated. If the dataset
    is larger than `n`, `n` random unique programs are sampled.
    """
    benchmarks = []
    with compiler_gym.make(env_name) as env:
        datasets = sorted(list(env.datasets))
        benchmarks_per_dataset = int(ceil(n / len(datasets)))

        for ds in datasets:
            logger.info(
                "Enumerating %s benchmarks from dataset from %s ...",
                benchmarks_per_dataset,
                ds,
            )
            if ds.size == 0 or ds.size > benchmarks_per_dataset:
                rng = np.random.default_rng(seed)
                uniq_bm_uris = set()
                benchmarks_from_dataset = []
                while len(benchmarks_from_dataset) < benchmarks_per_dataset:
                    bm = ds.random_benchmark(rng)
                    if bm.uri in uniq_bm_uris:
                        continue
                    uniq_bm_uris.add(bm.uri)
                    # Start an environment to check that the benchmark can be
                    # initialized.
                    try:
                        env.reset(benchmark=bm)
                    except (BenchmarkInitError, ValueError, TimeoutError):
                        continue
                    benchmarks_from_dataset.append(bm.uri)
                benchmarks += benchmarks_from_dataset
            else:
                bms = list(ds.benchmark_uris())
                bms *= int(ceil(benchmarks_per_dataset / len(bms)))
                benchmarks += bms[:benchmarks_per_dataset]

    benchmarks = sorted(benchmarks)
    with open(outdir / "benchmarks.txt", "w") as f:
        for bm in benchmarks:
            print(bm, file=f)
    return benchmarks


def chunkify(iterable, n):
    iterable = iter(iterable)
    chunk = list(islice(iterable, n))
    while chunk:
        yield chunk
        chunk = list(islice(iterable, n))


@app.command()
def reset(
    n: int = int(1e6),
    num_benchmarks: int = int(1e3),
    env: str = "llvm-autophase-ic-v0",
    j: int = cpu_count(),
    seed: int = 0xCC,
    outdir: Optional[Path] = None,
):
    """Benchmark the env.reset() operator."""
    executor = Executor(type="local", cpus=j)
    outdir = Path(outdir or create_user_logs_dir("op_benchmarks"))
    benchmarks = get_benchmarks(
        env_name=env, n=min(n, num_benchmarks), seed=seed, outdir=outdir
    )
    with executor.get_executor(logs_dir=outdir) as session:
        _reset(
            benchmarks=benchmarks,
            n=n,
            outdir=outdir,
            j=j,
            env_name=env,
            session=session,
        )


def _reset(
    benchmarks: List[str],
    n: int,
    outdir: Path,
    env_name: str,
    j: int,
    session: Executor,
):
    outdir.mkdir(exist_ok=True, parents=True)
    num_measurements_per_benchmark = int(ceil(n / len(benchmarks)))
    for i, benchmarks_chunk in enumerate(chunkify(benchmarks, j), start=1):
        session.submit(
            _reset_worker,
            num_measurements_per_benchmark=num_measurements_per_benchmark,
            benchmarks=benchmarks_chunk,
            env_name=env_name,
            outfile=outdir / f".op:2:reset-shard-{i:02d}.txt",
        )


def _reset_worker(
    num_measurements_per_benchmark: int,
    benchmarks: List[str],
    env_name: str,
    outfile: Path,
):
    with compiler_gym.make(env_name) as env:
        with open(outfile, "w") as f:
            for benchmark in benchmarks:
                env.reset(benchmark=benchmark)
                runtimes = get_runtimes(
                    lambda: env.reset(benchmark=benchmark),
                    num_measurements_per_benchmark,
                )
                print("\n".join(f"{x:.8f} {benchmark}" for x in runtimes), file=f)


@app.command()
def step(
    n: int = int(1e6),
    num_benchmarks: int = int(1e3),
    env: str = "llvm-autophase-ic-v0",
    j: int = cpu_count(),
    seed: int = 0xCC,
    outdir: Optional[Path] = None,
):
    """Benchmark the env.step() operator."""
    executor = Executor(type="local", cpus=j)
    outdir = Path(outdir or create_user_logs_dir("op_benchmarks"))
    benchmarks = get_benchmarks(
        env_name=env, n=min(n, num_benchmarks), seed=seed, outdir=outdir
    )
    with executor.get_executor(logs_dir=outdir) as session:
        _step(
            session=session,
            outdir=outdir,
            benchmarks=benchmarks,
            n=n,
            j=j,
            env_name=env,
            seed=seed,
        )


def _step(
    n: int,
    benchmarks: List[str],
    env_name: str,
    seed: int,
    j: int,
    outdir: Path,
    session: Executor,
):
    outdir.mkdir(exist_ok=True, parents=True)
    num_measurements_per_benchmark = int(ceil(n / len(benchmarks)))
    for i, benchmarks_chunk in enumerate(chunkify(benchmarks, j), start=1):
        session.submit(
            _step_worker,
            num_measurements_per_benchmark=num_measurements_per_benchmark,
            seed=seed + (i * len(benchmarks_chunk)),
            benchmarks=benchmarks_chunk,
            env_name=env_name,
            step_outfile=outdir / f".op:3:step-shard-{i:02d}.txt",
            batched_outfile=outdir / f".op:3:step-batched-shard-{i:02d}.txt",
        )


def _step_worker(
    num_measurements_per_benchmark: int,
    benchmarks: List[str],
    env_name: str,
    seed: str,
    step_outfile: Path,
    batched_outfile: Path,
):
    def get_step_times(env: CompilerEnv, num_steps: int, batched=False):
        while batched:
            # Run all actions in a single step().
            steps = [env.action_space.sample() for _ in range(num_steps)]
            with Timer() as timer:
                _, _, done, _ = env.multistep(steps)
            if not done:
                return [timer.time / num_steps] * num_steps
            env.reset()

        # Run each action as a step().
        runtimes = []
        while len(runtimes) < num_steps:
            with Timer() as timer:
                _, _, done, _ = env.step(env.action_space.sample())
            if done:
                env.reset()
            else:
                runtimes.append(timer.time)
        return runtimes

    with compiler_gym.make(env_name) as env:
        with open(step_outfile, "w") as f:
            for i, benchmark in enumerate(benchmarks, start=seed):
                env.reset(benchmark=benchmark)
                env.seed(i)
                runtimes = get_step_times(env, num_measurements_per_benchmark)
                print("\n".join(f"{x:.8f} {benchmark}" for x in runtimes), file=f)

        with open(batched_outfile, "w") as f:
            for i, benchmark in enumerate(benchmarks, start=seed):
                env.reset(benchmark=benchmark)
                env.seed(i)
                runtimes = get_step_times(
                    env, num_measurements_per_benchmark, batched=True
                )
                print("\n".join(f"{x:.8f} {benchmark}" for x in runtimes), file=f)


@app.command()
def observations(
    env: str = "llvm-autophase-ic-v0",
    observation_spaces: List[str] = [
        "Ir",
        "InstCount",
        "Autophase",
        "Inst2vec",
        "Programl",
        "IrInstructionCount",
        "ObjectTextSizeBytes",
        "Runtime",
    ],
    n: int = int(1e6),
    num_benchmarks: int = int(1e3),
    j: int = cpu_count(),
    seed: int = 0xCC,
    outdir: Optional[Path] = None,
) -> List[float]:
    """Benchmark the environment observation spaces."""
    executor = Executor(type="local", cpus=j)
    outdir = Path(outdir or create_user_logs_dir("op_benchmarks"))
    benchmarks = get_benchmarks(
        env_name=env, n=min(n, num_benchmarks), seed=seed, outdir=outdir
    )
    with executor.get_executor(logs_dir=outdir) as session:
        _observations(
            session=session,
            env_name=env,
            benchmarks=benchmarks,
            j=j,
            outdir=outdir,
            observation_spaces=observation_spaces,
            n=n,
        )


def _observations(
    observation_spaces: List[str],
    benchmarks: List[str],
    n: int,
    j: int,
    session: Executor,
    outdir: Path,
    env_name: str,
):
    outdir.mkdir(exist_ok=True, parents=True)
    num_measurements_per_benchmark = int(ceil(n / len(benchmarks)))
    for i, benchmarks_chunk in enumerate(chunkify(benchmarks, j), start=1):
        for observation_space in observation_spaces:
            session.submit(
                _observations_worker,
                observation_space=observation_space,
                num_measurements_per_benchmark=num_measurements_per_benchmark,
                benchmarks=benchmarks_chunk,
                env_name=env_name,
                outfile=outdir / f".observation:{observation_space}-shard-{i:02d}.txt",
            )


def _observations_worker(
    observation_space: str,
    num_measurements_per_benchmark: int,
    benchmarks: List[str],
    env_name: str,
    outfile: Path,
):
    with compiler_gym.make(env_name) as env:
        with open(outfile, "w") as f:
            for benchmark in benchmarks:
                env.reset(benchmark=benchmark)
                if "llvm-" in env_name and observation_space == "Runtime":
                    if not env.observation.IsRunnable():
                        return []
                    env.runtime_observation_count = 1
                    env.runtime_warmups_count = 0
                runtimes = get_runtimes(
                    lambda: env.observation[observation_space],
                    num_measurements_per_benchmark,
                )
                print("\n".join(f"{x:.8f}" for x in runtimes), file=f, flush=True)


@app.command()
def run(
    env: str = "llvm-autophase-ic-v0",
    observation_spaces: List[str] = [
        "Ir",
        "InstCount",
        "Autophase",
        "Inst2vec",
        "Programl",
        "IrInstructionCount",
        "ObjectTextSizeBytes",
        "Runtime",
    ],
    n: int = int(1e6),
    num_benchmarks: int = int(1e3),
    j: int = cpu_count(),
    outdir: Optional[Path] = None,
    seed: int = 0xCC,
):
    """Run all of the environment benchmarks."""
    executor = Executor(type="local", cpus=j)
    outdir = Path(outdir or create_user_logs_dir("op_benchmarks"))
    benchmarks = get_benchmarks(
        env_name=env, n=min(n, num_benchmarks), seed=seed, outdir=outdir
    )

    with executor.get_executor(logs_dir=outdir) as session:
        _init(env_name=env, session=session, j=j, n=n, outdir=outdir)
        _reset(
            benchmarks=benchmarks,
            n=n,
            outdir=outdir,
            j=j,
            env_name=env,
            session=session,
        )
        _step(
            n=n,
            j=j,
            benchmarks=benchmarks,
            env_name=env,
            seed=seed,
            outdir=outdir,
            session=session,
        )
        _observations(
            n=n,
            j=j,
            benchmarks=benchmarks,
            env_name=env,
            outdir=outdir,
            session=session,
            observation_spaces=observation_spaces,
        )

    info([outdir])


def _aggregate(
    root: Path, files: List[str], outfile: Path
) -> Optional[Dict[str, float]]:
    if not files:
        return
    if not (outfile).is_file():
        runtimes = []
        for file in files:
            with open(root / file) as f:
                runtimes += [float(x.split()[0]) for x in f if x.strip()]
        if not runtimes:
            return
        runtimes = np.sort(runtimes)
        with open(outfile, "w") as f:
            print("\n".join(map(str, sorted(runtimes))), file=f)
    else:
        with open(outfile) as f:
            runtimes = np.array(list(map(float, f)))
    return {
        "n": len(runtimes),
        "p50": np.median(runtimes),
        "p99": np.percentile(runtimes, 99),
        "mean": np.mean(runtimes),
    }


@app.command()
def info(outdirs: List[Path] = []):
    """Aggregate logs from previous runs."""
    outdirs = outdirs or ["~/logs/compiler_gym/op_benchmarks"]
    rows = []
    for outdir in outdirs:
        for root, _, files in os.walk(Path(outdir).expanduser()):
            root = Path(root)
            timestamp = "-".join([root.parent.name, root.name])

            shards = defaultdict(list)
            for file in files:
                match = re.match(r"\.([:\w-]+)-shard-\d+\.txt", file)
                if match:
                    shards[match.group(1)].append(file)

            for shard, files in shards.items():
                agg = _aggregate(root, files, root / f"{shard}.txt")
                if agg:
                    rows.append(
                        {
                            "timestamp": timestamp,
                            "op": shard,
                            **agg,
                        }
                    )

    df = pd.DataFrame(rows)
    df.sort_values(["op", "timestamp"], inplace=True)

    # Scale to milliseconds.
    df["p50"] *= 1000
    df["p99"] *= 1000
    df["mean"] *= 1000
    df = df.rename(columns={"p50": "p50 (ms)", "p99": "p99 (ms)", "mean": "mean (ms)"})

    print(tabulate(df, headers="keys", showindex=False, tablefmt="psql"))


if __name__ == "__main__":
    init_logging()
    app()

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
from pathlib import Path
from typing import List

import pandas as pd
from llvm_autotuning.experiment import Experiment
from pydantic import ValidationError
from tabulate import tabulate
from typer import Typer

from compiler_gym.util.statistics import geometric_mean

app = Typer()


def experiments_from_paths(log_dirs: List[Path]) -> List[Experiment]:
    experiments: List[Experiment] = []
    for path in log_dirs:
        try:
            experiments += Experiment.from_logsdir(Path(path).expanduser())
        except ValidationError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    return experiments


@app.command()
def info(
    log_dirs: List[Path] = ["~/logs/compiler_gym/llvm_autotuning"],
    all_runs: bool = False,
    group_by_working_directory: bool = False,
    only_nonzero_reward: bool = False,
):
    experiments = experiments_from_paths(log_dirs)

    results = []
    for experiment in experiments:
        df = experiment.dataframe

        # Exclude runs where reward was zero, used for pruning false results if
        # the environment is flaky or can fail.
        if only_nonzero_reward:
            df = df[df.reward != 0]

        if not len(df):
            continue

        df.to_csv(experiment.working_directory / "results.csv", index=False)

        walltimes = df[["benchmark", "walltime"]].groupby("benchmark").mean()
        rewards = df[["benchmark", "reward"]].groupby("benchmark").agg(geometric_mean)
        num_results = len(df)
        num_benchmarks = len(set(df["benchmark"]))

        df = pd.concat((walltimes, rewards), axis=1)
        avg_walltime = df["walltime"].mean()
        avg_reward = geometric_mean(df["reward"])
        df = pd.concat(
            (
                df,
                pd.DataFrame(
                    [{"walltime": avg_walltime, "reward": avg_reward}],
                    index=["Average"],
                ),
            )
        )

        df = df.reset_index()
        df.insert(0, "config", experiment.configuration_number)
        df.insert(0, "timestamp", experiment.timestamp)
        df.insert(0, "experiment", experiment.experiment)

        if all_runs:
            print(experiment.working_directory)
            print(tabulate(df, showindex=False, headers="keys", tablefmt="grid"))
            print()

        results.append(
            {
                "working_directory": experiment.working_directory,
                "experiment": experiment.experiment,
                "timestamp": experiment.timestamp,
                "config": experiment.configuration_number,
                "num_benchmarks": num_benchmarks,
                "num_results": num_results,
                "walltime": avg_walltime,
                "reward": avg_reward,
            }
        )

    df = pd.DataFrame(results)
    if not len(df):
        print("No results")
        return
    print("---------------------------------------")
    print("Aggregate over experiments:")
    if group_by_working_directory:
        df = df.groupby(["working_directory"]).mean()
    else:
        df = df.groupby(["experiment", "timestamp", "config"]).mean()

    # Cast float back to int.
    df["num_benchmarks"] = [int(x) for x in df["num_benchmarks"]]
    df["num_results"] = [int(x) for x in df["num_results"]]

    # Better column names.
    df = df.rename(columns={"reward": "geomean_reward", "walltime": "walltime (s)"})

    pd.set_option("display.max_rows", None)
    print(df)


if __name__ == "__main__":
    app()

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
import sys
from pathlib import Path
from typing import List

import humanize
import pandas as pd
from llvm_rl.model import Model
from pydantic import ValidationError
from tabulate import tabulate
from typer import Typer

from compiler_gym.util.logging import init_logging
from compiler_gym.util.statistics import geometric_mean

app = Typer()


def models_from_paths(log_dirs: List[Path]):
    # Read all the inputs first.
    models: List[Model] = []
    for path in log_dirs:
        try:
            models += Model.from_logsdir(Path(path).expanduser())
        except ValidationError as e:
            print(e, file=sys.stderr)
            sys.exit(1)
    return models


@app.command()
def train(log_dirs: List[Path] = ["~/logs/compiler_gym/llvm_rl"]):
    init_logging()

    models = models_from_paths(log_dirs)

    dfs = []
    for model in models:
        df = model.dataframe
        if not len(df):
            continue

        # Select only the rows with a checkpoint.
        df = df[df["checkpoint"].values]

        df = df[
            [
                "trial_name",
                "experiment_timestamp",
                "episodes_total",
                "episode_reward_geomean",
                "episode_reward_mean",
                "evaluation/episode_reward_mean",
                "evaluation/episode_reward_geomean",
                "time_total_s",
                "complete",
                "cpus",
                "gpus",
            ]
        ]

        sdf = df.groupby(
            ["experiment", "config", "replica", "experiment_timestamp"]
        ).max()

        test_results = model.test_dataframes
        sdf["test_results"] = [
            test_results.get(d, pd.DataFrame()) for d in sdf["trial_name"]
        ]

        sdf["test_ic_mean"] = [
            sum(d["instruction_count_reduction"]) / len(d)
            if not d.empty
            else float("nan")
            for d in sdf["test_results"]
        ]
        sdf["test_ic_geomean"] = [
            geometric_mean(d["instruction_count_reduction"])
            if not d.empty
            else float("nan")
            for d in sdf["test_results"]
        ]
        sdf["test_os_mean"] = [
            sum(d["object_size_reduction"]) / len(d) if not d.empty else float("nan")
            for d in sdf["test_results"]
        ]
        sdf["test_os_geomean"] = [
            geometric_mean(d["object_size_reduction"]) if not d.empty else float("nan")
            for d in sdf["test_results"]
        ]
        sdf["test_checkpoint"] = [
            int(d["test_checkpoint"].values[0].split("-")[-1]) if not d.empty else ""
            for d in sdf["test_results"]
        ]

        dfs.append(sdf.reset_index())

    df = pd.concat(dfs)

    # Print everything.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    df = df.rename(
        columns={
            "experiment_timestamp": "timestamp",
            "episodes_total": "episodes",
            "evaluation/episode_reward_geomean": "val_geomean",
            "evaluation/episode_reward_mean": "val_mean",
            "episode_reward_mean": "train_mean",
            "episode_reward_geomean": "train_geomean",
            "time_total_s": "training_time",
            "test_reward_mean": "test_mean",
            "test_reward_geomean": "test_geomean",
        }
    )

    # Format for printing.
    df["complete"] = [f"{x:.1%}" for x in df["complete"]]
    df["episodes"] = [f"{int(x):,d}" for x in df["episodes"]]
    df["training_time"] = [humanize.naturaldelta(x) for x in df["training_time"]]

    for reward in [
        "train_mean",
        "train_geomean",
        "val_mean",
        "val_geomean",
        "test_ic_geomean",
        "test_os_geomean",
        "test_ic_mean",
        "test_os_mean",
    ]:
        df[reward] = [f"{x:.4f}" for x in df[reward].values]

    df = df[
        [
            "trial_name",
            "timestamp",
            "complete",
            "episodes",
            "training_time",
            "test_checkpoint",
            "train_geomean",
            "val_geomean",
        ]
    ]

    print(tabulate(df, headers="keys", showindex=False, tablefmt="psql"))


@app.command()
def test(
    log_dirs: List[Path] = ["~/logs/compiler_gym/llvm_rl"],
    format_for_latex: bool = False,
):
    models = models_from_paths(log_dirs)

    # Print everything.
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", None)

    dfs = {}
    for model in models:
        for trial, df in model.test_dataframes.items():
            df["test_set"] = [
                re.search(r"^((benchmark|generator)://)(.+)-v[012]/", d).group(3)
                for d in df["benchmark"]
            ]

            # Prune empty test set.
            df = df[df["instruction_count_init"] > 0]

            gmean_df = (
                df[
                    [
                        "test_set",
                        "instruction_count_reduction",
                        "object_size_reduction",
                    ]
                ]
                .groupby(["test_set"])
                .agg(geometric_mean)
            )

            mean_df = (
                df[
                    [
                        "test_set",
                        "inference_walltime_seconds",
                    ]
                ]
                .groupby(["test_set"])
                .mean()
            )

            df = pd.concat((mean_df, gmean_df), axis=1)
            df = df.reset_index()
            df.insert(0, "trial", trial)

            if format_for_latex:
                df["instruction_count_reduction"] = [
                    f"${float(d):.3f}\\times$"
                    for d in df["instruction_count_reduction"]
                ]
                df["object_size_reduction"] = [
                    f"${float(d):.3f}\\times$" for d in df["object_size_reduction"]
                ]

            print()
            print(tabulate(df, headers="keys", showindex=False, tablefmt="psql"))

            dfs[trial] = df


if __name__ == "__main__":
    app()

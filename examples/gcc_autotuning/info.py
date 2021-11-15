# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
from llvm_autotuning.experiment import Experiment
from pydantic import ValidationError
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
    log_dirs: List[Path] = ["~/logs/compiler_gym/gcc_autotuning"],
):
    dfs: List[pd.DataFrame] = []

    for path in log_dirs:
        path = Path(path).expanduser()
        for root, _, files in os.walk(path):
            if "results.csv" not in files:
                continue

            root = Path(root)

            df = pd.read_csv(root / "results.csv")
            if not df.size:
                continue

            df["timestamp"] = "-".join([root.parent.name, root.name])

            dfs.append(df)

    if not dfs:
        print("No results")

    df = pd.concat(dfs)
    df = df.groupby(["timestamp", "search"])[["scaled_size"]].agg(geometric_mean)
    df = df.rename(columns={"scaled_size": "geomean_reward"})

    pd.set_option("display.max_rows", None)
    print(df)


if __name__ == "__main__":
    app()

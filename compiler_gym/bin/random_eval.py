# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Evaluate the logs of a random run."""
import json
from pathlib import Path

import humanize
import numpy as np
from absl import app, flags

import compiler_gym.util.flags.output_dir  # noqa Flag definition.
from compiler_gym.util import logs
from compiler_gym.util.statistics import geometric_mean
from compiler_gym.util.tabulate import tabulate

FLAGS = flags.FLAGS


def eval_logs(outdir: Path) -> None:
    rows = []
    totals = {
        "instructions": 0,
        "init_reward": [],
        "max_reward": [],
        "attempts": 0,
        "time": 0,
        "actions": 0,
    }

    for results_dir in sorted(outdir.iterdir()):
        benchmark = results_dir.name
        progress_path = results_dir / logs.PROGRESS_LOG_NAME
        meta_path = results_dir / logs.METADATA_NAME

        if (
            not results_dir.is_dir()
            or not progress_path.is_file()
            or not meta_path.is_file()
        ):
            continue

        with open(meta_path, "rb") as f:
            meta = json.load(f)

        with open(str(progress_path)) as f:
            final_line = f.readlines()[-1]
        best = logs.ProgressLogEntry.from_csv(final_line)

        totals["instructions"] += meta["num_instructions"]
        totals["init_reward"].append(meta["init_reward"])
        totals["max_reward"].append(best.reward)
        totals["attempts"] += best.total_episode_count
        totals["time"] += best.runtime_seconds
        totals["actions"] += best.num_passes

        rows.append(
            (
                benchmark,
                humanize.intcomma(meta["num_instructions"]),
                f"{meta['init_reward']:.4f}",
                f"{best.reward:.4f}",
                (
                    f"{humanize.intcomma(best.total_episode_count)} attempts "
                    f"in {humanize.naturaldelta(best.runtime_seconds)}"
                ),
                humanize.intcomma(best.num_passes),
            )
        )

    row_count = len(totals["init_reward"])
    rows.append(
        (
            "Geomean",
            "",
            f"{geometric_mean(totals['init_reward']):.4f}",
            f"{geometric_mean(totals['max_reward']):.4f}",
            "",
            "",
        )
    )
    rows.append(
        (
            "Average",
            humanize.intcomma(int(totals["instructions"] / row_count)),
            f"{np.array(totals['init_reward']).mean():.4f}",
            f"{np.array(totals['max_reward']).mean():.4f}",
            (
                f"{humanize.intcomma(int(totals['attempts'] / row_count))} attempts "
                f"in {humanize.naturaldelta(totals['time'] / row_count)}"
            ),
            humanize.intcomma(int(totals["actions"] / row_count)),
        )
    )

    print(
        tabulate(
            rows,
            headers=(
                "Benchmark",
                "#. instructions",
                "Init Reward",
                "Max Reward",
                "Found after",
                "#. actions",
            ),
        )
    )


def main(argv):
    """Main entry point."""
    argv = FLAGS(argv)
    if len(argv) != 1:
        raise app.UsageError(f"Unknown command line arguments: {argv[1:]}")

    output_dir = Path(FLAGS.output_dir).expanduser().resolve().absolute()
    assert output_dir.is_dir(), f"Directory not found: {output_dir}"

    eval_logs(output_dir)


if __name__ == "__main__":
    app.run(main)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a class to represent a compiler environment state."""
import csv
import sys
from typing import Iterable, List, Optional, TextIO

from pydantic import BaseModel, Field, validator

from compiler_gym.datasets.uri import BENCHMARK_URI_PATTERN
from compiler_gym.util.truncate import truncate


class CompilerEnvState(BaseModel):
    """The representation of a compiler environment state.

    The state of an environment is defined as a benchmark and a sequence of
    actions that has been applied to it. For a given environment, the state
    contains the information required to reproduce the result.
    """

    benchmark: str = Field(
        allow_mutation=False,
        regex=BENCHMARK_URI_PATTERN,
        examples=[
            "benchmark://cbench-v1/crc32",
            "generator://csmith-v0/0",
        ],
    )
    """The URI of the benchmark used for this episode."""

    commandline: str
    """The list of actions that produced this state, as a commandline."""

    walltime: float
    """The walltime of the episode in seconds. Must be non-negative."""

    reward: Optional[float] = Field(
        required=False,
        default=None,
        allow_mutation=True,
    )
    """The cumulative reward for this episode. Optional."""

    @validator("walltime")
    def walltime_nonnegative(cls, v):
        if v is not None:
            assert v >= 0, "Walltime cannot be negative"
        return v

    @property
    def has_reward(self) -> bool:
        """Return whether the state has a reward value."""
        return self.reward is not None

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, CompilerEnvState):
            return False
        epsilon = 1e-5
        # Only compare reward if both states have it.
        if not (self.has_reward and rhs.has_reward):
            reward_equal = True
        else:
            reward_equal = abs(self.reward - rhs.reward) < epsilon
        # Note that walltime is excluded from equivalence checks as two states
        # are equivalent if they define the same point in the optimization space
        # irrespective of how long it took to get there.
        return (
            self.benchmark == rhs.benchmark
            and reward_equal
            and self.commandline == rhs.commandline
        )

    def __ne__(self, rhs) -> bool:
        return not self == rhs

    class Config:
        validate_assignment = True


class CompilerEnvStateWriter:
    """Serialize compiler environment states to CSV.

    Example use:

        >>> with CompilerEnvStateWriter(open("results.csv", "wb")) as writer:
        ...     writer.write_state(env.state)
    """

    def __init__(self, f: TextIO, header: bool = True):
        """Constructor.

        :param f: The file to write to.
        :param header: Whether to include a header row.
        """
        self.f = f
        self.writer = csv.writer(self.f, lineterminator="\n")
        self.header = header

    def write_state(self, state: CompilerEnvState, flush: bool = False) -> None:
        """Write the state to file.

        :param state: A compiler environment state.

        :param flush: Write to file immediately.
        """
        if self.header:
            self.writer.writerow(("benchmark", "reward", "walltime", "commandline"))
            self.header = False
        self.writer.writerow(
            (state.benchmark, state.reward, state.walltime, state.commandline)
        )
        if flush:
            self.f.flush()

    def __enter__(self):
        """Support with-statement for the writer."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the writer."""
        self.f.close()


class CompilerEnvStateReader:
    """Read states from a CSV file.

    Example usage:

        >>> with CompilerEnvStateReader(open("results.csv", "rb")) as reader:
        ...     for state in reader:
        ...         print(state)
    """

    def __init__(self, f: TextIO):
        """Constructor.

        :param f: The file to read.
        """
        self.f = f
        self.reader = csv.reader(self.f)

    def __iter__(self) -> Iterable[CompilerEnvState]:
        """Read the states from the file."""
        columns_in_order = ["benchmark", "reward", "walltime", "commandline"]
        # Read the CSV and coerce the columns into the expected order.
        for (
            benchmark,
            reward,
            walltime,
            commandline,
        ) in self._iterate_columns_in_order(self.reader, columns_in_order):
            yield CompilerEnvState(
                benchmark=benchmark,
                reward=None if reward == "" else float(reward),
                walltime=0 if walltime == "" else float(walltime),
                commandline=commandline,
            )

    @staticmethod
    def _iterate_columns_in_order(
        reader: csv.reader, columns: List[str]
    ) -> Iterable[List[str]]:
        """Read the input CSV and return each row in the given column order.

        Supports CSVs both with and without a header. If no header, columns are
        expected to be in the correct order. Else the header row is used to
        determine column order.

        Header row detection is case insensitive.

        :param reader: The CSV file to read.

        :param columns: A list of column names in the order that they are
            expected.

        :return: An iterator over rows.
        """
        try:
            row = next(reader)
        except StopIteration:
            # Empty file.
            return

        if len(row) != len(columns):
            raise ValueError(
                f"Expected {len(columns)} columns in the first row of CSV: {truncate(row)}"
            )

        # Convert the maybe-header columns to lowercase for case-insensitive
        # comparison.
        maybe_header = [v.lower() for v in row]
        if set(maybe_header) == set(columns):
            # The first row matches the expected columns names, so use it to
            # determine the column order.
            column_order = [maybe_header.index(v) for v in columns]
            yield from ([row[v] for v in column_order] for row in reader)
        else:
            # The first row isn't a header, so assume that all rows are in
            # expected column order.
            yield row
            yield from reader

    def __enter__(self):
        """Support with-statement for the reader."""
        return self

    def __exit__(self, *args):
        """Support with-statement for the reader."""
        self.f.close()

    @staticmethod
    def read_paths(paths: Iterable[str]) -> Iterable[CompilerEnvState]:
        """Read a states from a list of file paths.

        Read states from stdin using a special path :code:`"-"`.

        :param: A list of paths.

        :return: A generator of compiler env states.
        """
        for path in paths:
            if path == "-":
                yield from iter(CompilerEnvStateReader(sys.stdin))
            else:
                with open(path) as f:
                    yield from iter(CompilerEnvStateReader(f))

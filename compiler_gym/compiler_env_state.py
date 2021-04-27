# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a class to represent a compiler environment state."""
import csv
from io import StringIO
from typing import Iterable, Optional

from pydantic import BaseModel, Field
from pydantic import ValidationError as PydanticValidationError
from pydantic import validator

from compiler_gym.datasets.uri import BENCHMARK_URI_PATTERN


def _to_csv(*columns) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    return buf.getvalue().rstrip()


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
    """The walltime of the episode in seconds. Must be nonnegative. Optional."""

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

    @staticmethod
    def csv_header() -> str:
        """Return the header string for the CSV-format.

        :return: A comma-separated string.
        """
        return _to_csv("benchmark", "reward", "walltime", "commandline")

    def to_csv(self) -> str:
        """Serialize a state to a comma separated list of values.

        :return: A comma-separated string.
        """
        return _to_csv(self.benchmark, self.reward, self.walltime, self.commandline)

    @classmethod
    def from_csv(cls, csv_string: str) -> "CompilerEnvState":
        """Construct a state from a comma separated list of values."""
        reader = csv.reader(StringIO(csv_string))
        for line in reader:
            try:
                benchmark, reward, walltime, commandline = line
                break
            except (ValueError, PydanticValidationError) as e:
                raise ValueError(f"Failed to parse input: `{csv_string}`: {e}") from e
        else:
            raise ValueError(f"Failed to parse input: `{csv_string}`")
        return cls(
            benchmark=benchmark,
            reward=None if reward == "" else float(reward),
            walltime=0 if walltime == "" else float(walltime),
            commandline=commandline,
        )

    @classmethod
    def read_csv_file(cls, in_file) -> Iterable["CompilerEnvState"]:
        """Read states from a CSV file.

        :param in_file: A file object.
        :returns: A generator of :class:`CompilerEnvState` instances.
        :raises ValueError: If input parsing fails.
        """
        # TODO(cummins): Check schema of DictReader and, on failure, fallback
        # to from_csv() per-line.
        # TODO(cummins): Accept a URL for in_file and read from web.
        data = in_file.readlines()
        for line in csv.DictReader(data):
            try:
                line["reward"] = float(line["reward"]) if line.get("reward") else None
                line["walltime"] = (
                    float(line["walltime"]) if line.get("walltime") else None
                )
                yield CompilerEnvState(**line)
            except (TypeError, KeyError, PydanticValidationError) as e:
                raise ValueError(f"Failed to parse input: `{e}`") from e

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

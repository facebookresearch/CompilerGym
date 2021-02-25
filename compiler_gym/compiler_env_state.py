# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a class to represent a compiler environment state."""
import csv
from io import StringIO
from typing import Any, Dict, Iterable, NamedTuple, Optional


def _to_csv(*columns) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    return buf.getvalue().rstrip()


class CompilerEnvState(NamedTuple):
    """The representation of a compiler environment state.

    The state of an environment is defined as a benchmark and a sequence of
    actions that has been applied to it. For a given environment, the state
    contains the information required to reproduce the result.
    """

    benchmark: str
    """The name of the benchmark used for this episode."""

    commandline: str
    """The list of actions that produced this state, as a commandline."""

    walltime: float
    """The walltime of the episode."""

    reward: Optional[float] = None
    """The cumulative reward for this episode."""

    @staticmethod
    def csv_header() -> str:
        """Return the header string for the CSV-format.

        :return: A comma-separated string.
        """
        return _to_csv("benchmark", "reward", "walltime", "commandline")

    def json(self):
        """Return the state as JSON."""
        return self._asdict()  # pylint: disable=no-member

    def to_csv(self) -> str:
        """Serialize a state to a comma separated list of values.

        :return: A comma-separated string.
        """
        return _to_csv(self.benchmark, self.reward, self.walltime, self.commandline)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "CompilerEnvState":
        """Construct a state from a JSON dictionary."""
        return cls(**data)

    @classmethod
    def from_csv(cls, csv_string: str) -> "CompilerEnvState":
        """Construct a state from a comma separated list of values."""
        reader = csv.reader(StringIO(csv_string))
        for line in reader:
            try:
                benchmark, reward, walltime, commandline = line
                break
            except ValueError as e:
                raise ValueError(f"Failed to parse input: `{csv_string}`: {e}") from e
        else:
            raise ValueError(f"Failed to parse input: `{csv_string}`")
        return cls(
            benchmark=benchmark,
            reward=None if reward == "" else float(reward),
            walltime=float(walltime),
            commandline=commandline,
        )

    @classmethod
    def read_csv_file(cls, in_file) -> Iterable["CompilerEnvState"]:
        """Read states from a CSV file.

        :param in_file: A file object.
        :returns: A generator of :class:`CompilerEnvState` instances.
        :raises ValueError: If input parsing fails.
        """
        data = in_file.readlines()
        for line in csv.DictReader(data):
            try:
                line["reward"] = float(line["reward"]) if line.get("reward") else None
                line["walltime"] = (
                    float(line["walltime"]) if line.get("walltime") else None
                )
                yield CompilerEnvState(**line)
            except (TypeError, KeyError) as e:
                raise ValueError(f"Failed to parse input: `{e}`") from e

    def __eq__(self, rhs) -> bool:
        if not isinstance(rhs, CompilerEnvState):
            return False
        epsilon = 1e-5
        # If only one benchmark has a reward the states cannot be equal.
        if (self.reward is None) != (rhs.reward is None):
            return False
        if (self.reward is None) and (rhs.reward is None):
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

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the validation result tuple."""
import re
from typing import NamedTuple

from compiler_gym.compiler_env_state import CompilerEnvState


class ValidationResult(NamedTuple):
    """A tuple that represents the result of validating a compiler environment state."""

    state: CompilerEnvState
    """The compiler environment state that was validated."""

    walltime: float
    """The wall time in seconds that the validation took."""

    reward_validated: bool = False
    """Whether the reward that was recorded in the original state was validated."""

    actions_replay_failed: bool = False
    """Whether the commandline was unable to be reproduced."""

    reward_validation_failed: bool = False
    """Whether the validated reward differed from the original state."""

    benchmark_semantics_validated: bool = False
    """Whether the semantics of the benchmark were validated."""

    benchmark_semantics_validation_failed: bool = False
    """Whether the semantics of the benchmark were found to have changed."""

    error_details: str = ""
    """A description of any validation errors."""

    def okay(self) -> bool:
        """Whether validation succeeded."""
        return not (
            self.actions_replay_failed
            or self.reward_validation_failed
            or self.benchmark_semantics_validation_failed
        )

    def __repr__(self):
        # Remove default-protocol prefix to improve output readability.
        benchmark = re.sub(r"^benchmark://", "", self.state.benchmark)

        if not self.okay():
            msg = ", ".join(self.error_details.strip().split("\n"))
            return f"❌  {benchmark}  {msg}"
        elif self.state.reward is None:
            return f"✅  {benchmark}"
        else:
            return f"✅  {benchmark}  {self.state.reward:.4f}"

    def json(self):
        """Get the state as a JSON-serializable dictionary.

        :return: A JSON dict.
        """
        data = self._asdict()  # pylint: disable=no-member
        data["state"] = self.state.json()
        return data

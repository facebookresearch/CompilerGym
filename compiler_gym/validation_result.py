# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines the validation result tuple."""
import itertools
import re
from collections import Counter
from typing import Iterable, List

from pydantic import BaseModel, validator

from compiler_gym.compiler_env_state import CompilerEnvState
from compiler_gym.util.shell_format import plural
from compiler_gym.util.truncate import truncate
from compiler_gym.validation_error import ValidationError


class ValidationResult(BaseModel):
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

    errors: List[ValidationError] = []
    """A list of :class:`ValidationError <compiler_gym.ValidationError>` """

    @validator("walltime")
    def walltime_nonnegative(cls, v):
        assert v >= 0, "Walltime cannot be negative"
        return v

    def __eq__(self, rhs):
        """Equality comparison.

        Validation results are *not* compared on walltime, and are insensitive
        to the order of errors.
        """
        if not isinstance(rhs, ValidationResult):
            return False
        return (
            self.state == rhs.state
            and self.reward_validated == rhs.reward_validated
            and self.actions_replay_failed == rhs.actions_replay_failed
            and self.reward_validation_failed == rhs.reward_validation_failed
            and self.benchmark_semantics_validated == rhs.benchmark_semantics_validated
            and self.benchmark_semantics_validation_failed
            == rhs.benchmark_semantics_validation_failed
            and sorted(self.errors) == sorted(rhs.errors)
        )

    def __ne__(self, rhs):
        return not self == rhs

    @property
    def error_details(self) -> str:
        """A summary description of the validation errors."""
        if not self.errors:
            return ""

        msg = []
        error_types = [e.type for e in self.errors]
        freq = sorted(Counter(error_types).items(), key=lambda x: -x[1])

        # Shortcut for when there is just a single message to aggregate. Use
        # format: "${error_msg}" if there is a single error or "${n}×
        # ${error_msg}" if there are multiple copies of the same error.
        if len(freq) == 1:
            message = str(error_types[0])
            if len(error_types) == 1:
                return message
            return f"{len(error_types)}× {message}"

        # If there are multiple error messages, number them using the format:
        # "[${i}/${j}] ${n}× ${error_msg}". E.g. "[1/3] 18× Memory leak".
        for j, (message, count) in enumerate(freq, start=1):
            if count > 1:
                msg.append(f"[{j}/{len(freq)}] {count}× {message}")
            else:
                msg.append(f"[{j}/{len(freq)}] {message}")
            remaining = len(freq) - j
            if j >= 3 and remaining > 3:
                msg.append(
                    f"... ({remaining} more {plural(remaining, 'error', 'errors')})"
                )
                break
        return ", ".join(msg)

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
            return f"❌  {benchmark}  {truncate(msg, max_lines=1, max_line_len=50)}"
        elif self.state.reward is None:
            return f"✅  {benchmark}"
        else:
            return f"✅  {benchmark}  {self.state.reward:.4f}"

    def __str__(self):
        return repr(self)

    @classmethod
    def join(cls, results: Iterable["ValidationResult"]):
        """Create a validation result that is the union join of multiple results."""
        results = list(results)
        if not results:
            raise ValueError("No states to join")
        if any(r.state != results[0].state for r in results[1:]):
            raise ValueError("All states must be the same")

        return cls(
            # NOTE: No checking that states are the same.
            state=results[0].state,
            walltime=sum(r.walltime for r in results),
            reward_validated=any(r.reward_validated for r in results),
            actions_replay_failed=any(r.actions_replay_failed for r in results),
            reward_validation_failed=any(r.reward_validation_failed for r in results),
            benchmark_semantics_validated=any(
                r.benchmark_semantics_validated for r in results
            ),
            benchmark_semantics_validation_failed=any(
                r.benchmark_semantics_validation_failed for r in results
            ),
            errors=list(itertools.chain.from_iterable(r.errors for r in results)),
        )

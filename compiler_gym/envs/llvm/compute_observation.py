# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a utility function for computing LLVM observations."""
import subprocess
import sys
from pathlib import Path
from typing import List

import google.protobuf.text_format

from compiler_gym.service.proto import Observation
from compiler_gym.util.gym_type_hints import ObservationType
from compiler_gym.util.runfiles_path import runfiles_path
from compiler_gym.util.shell_format import plural
from compiler_gym.views.observation_space_spec import ObservationSpaceSpec

_COMPUTE_OBSERVATION_BIN = runfiles_path(
    "compiler_gym/envs/llvm/service/compute_observation"
)


def pascal_case_to_enum(pascal_case: str) -> str:
    """Convert PascalCase to ENUM_CASE."""
    word_arrays: List[List[str]] = [[]]

    for c in pascal_case:
        if c.isupper() and word_arrays[-1]:
            word_arrays.append([c])
        else:
            word_arrays[-1].append(c.upper())

    return "_".join(["".join(word) for word in word_arrays])


def compute_observation(
    observation_space: ObservationSpaceSpec, bitcode: Path, timeout: float = 300
) -> ObservationType:
    """Compute an LLVM observation.

    This is a utility function that uses a standalone C++ binary to compute an
    observation from an LLVM bitcode file. It is intended for use cases where
    you want to compute an observation without the overhead of initializing a
    full environment.

    Example usage:

        >>> env = compiler_gym.make("llvm-v0")
        >>> space = env.observation.spaces["Ir"]
        >>> bitcode = Path("bitcode.bc")
        >>> observation = llvm.compute_observation(space, bitcode, timeout=30)

    .. warning::

        This is not part of the core CompilerGym API and may change in a future
        release.

    :param observation_space: The observation that is to be computed.

    :param bitcode: The path of an LLVM bitcode file.

    :param timeout: The maximum number of seconds to allow the computation to
        run before timing out.

    :raises ValueError: If computing the observation fails.

    :raises TimeoutError: If computing the observation times out.

    :raises FileNotFoundError: If the given bitcode does not exist.
    """
    if not Path(bitcode).is_file():
        raise FileNotFoundError(bitcode)

    observation_space_name = pascal_case_to_enum(observation_space.id)

    process = subprocess.Popen(
        [str(_COMPUTE_OBSERVATION_BIN), observation_space_name, str(bitcode)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as e:
        # kill() was added in Python 3.7.
        if sys.version_info >= (3, 7, 0):
            process.kill()
        else:
            process.terminate()
        process.communicate(timeout=timeout)  # Wait for shutdown to complete.
        raise TimeoutError(
            f"Failed to compute {observation_space.id} observation in "
            f"{timeout:.1f} {plural(int(round(timeout)), 'second', 'seconds')}"
        ) from e

    if process.returncode:
        try:
            stderr = stderr.decode("utf-8")
            raise ValueError(
                f"Failed to compute {observation_space.id} observation: {stderr}"
            )
        except UnicodeDecodeError as e:
            raise ValueError(
                f"Failed to compute {observation_space.id} observation"
            ) from e

    try:
        stdout = stdout.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(
            f"Failed to parse {observation_space.id} observation: {e}"
        ) from e

    observation = Observation()
    try:
        google.protobuf.text_format.Parse(stdout, observation)
    except google.protobuf.text_format.ParseError as e:
        raise ValueError(f"Failed to parse {observation_space.id} observation") from e

    return observation_space.translate(observation)

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for LexedIRTuple derived observation space."""
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, List, NamedTuple

import google.protobuf.text_format
import numpy as np

from compiler_gym.service.proto import Event
from compiler_gym.service.proto.py_converters import make_message_default_converter
from compiler_gym.util.commands import Popen
from compiler_gym.util.runfiles_path import runfiles_path
from compiler_gym.util.shell_format import plural

_COMPUTE_OBSERVATION_BIN = runfiles_path(
    "compiler_gym/envs/llvm/service/compute_observation"
)
_COMPUTE_UNLEX_BIN = runfiles_path("compiler_gym/third_party/Lexedir/compute_unlexed")


class LexedToken(NamedTuple):
    ID: int
    kind: str
    category: str
    value: str


def LexedIr(bitcode: Path, timeout: float = 300) -> Dict[str, np.array]:
    """ """
    if not Path(bitcode).is_file():
        raise FileNotFoundError(bitcode)

    observation_space_name = "LEXED_IR"
    translate = make_message_default_converter()

    try:
        with Popen(
            [str(_COMPUTE_OBSERVATION_BIN), observation_space_name, str(bitcode)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as process:
            stdout, stderr = process.communicate(timeout=timeout)

            if process.returncode:
                try:
                    stderr = stderr.decode("utf-8")
                    raise ValueError(f"Failed to compute LexedIr observation: {stderr}")
                except UnicodeDecodeError as e:
                    raise ValueError("Failed to compute LexedIr observation") from e
    except subprocess.TimeoutExpired as e:
        raise TimeoutError(
            "Failed to compute LexedIr observation in "
            f"{timeout:.1f} {plural(int(round(timeout)), 'second', 'seconds')}"
        ) from e

    try:
        stdout = stdout.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ValueError(f"Failed to parse LexedIr observation: {e}") from e

    observation = Event()
    try:
        google.protobuf.text_format.Parse(stdout, observation)
    except google.protobuf.text_format.ParseError as e:
        raise ValueError("Failed to parse LexedIr observation") from e

    return translate(observation)


def LexedIrTuple(bitcode: Path, timeout: float = 300) -> List[LexedToken]:
    """
    Standalone IR Lexer.
    """
    lexed_dict = LexedIr(bitcode, timeout=timeout)
    return [
        LexedToken(tid, tval, tkind, tcat)
        for tid, tval, tkind, tcat in zip(
            lexed_dict["token_id"],
            lexed_dict["token_value"],
            lexed_dict["token_kind"],
            lexed_dict["token_category"],
        )
    ]


def UnLex(token_ids: List[int], token_values: List[str], timeout: float = 300) -> str:

    with NamedTemporaryFile("w", prefix="compiler_gym_unlex_") as f:
        f.write(
            "\n".join(["{},{}".format(i, v) for i, v in zip(token_ids, token_values)])
        )
        f.flush()
        try:
            with Popen(
                [str(_COMPUTE_UNLEX_BIN), str(f.name)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            ) as process:
                stdout, stderr = process.communicate(timeout=timeout)

                if process.returncode:
                    try:
                        stderr = stderr.decode("utf-8")
                        raise ValueError(
                            f"Failed to compute UnLex observation: {stderr}"
                        )
                    except UnicodeDecodeError as e:
                        raise ValueError("Failed to compute UnLex observation") from e
        except subprocess.TimeoutExpired as e:
            raise TimeoutError(
                f"Failed to compute UnLex observation in "
                f"{timeout:.1f} {plural(int(round(timeout)), 'second', 'seconds')}"
            ) from e

        try:
            stdout = stdout.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ValueError(f"Failed to parse UnLex observation: {e}") from e

    return stdout

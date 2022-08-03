# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for LexedIRTuple derived observation space."""
import typing


class LexedToken(typing.NamedTuple):
    ID: int
    kind: str
    category: str
    value: str

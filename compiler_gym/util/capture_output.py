# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import sys
from io import StringIO
from typing import Iterator


class CapturedOutput:
    def __init__(self):
        self.stdout = StringIO()
        self.stderr = StringIO()


@contextlib.contextmanager
def capture_output() -> Iterator[CapturedOutput]:
    """Context manager to temporarily capture stdout/stderr."""
    stdout, stderr = sys.stdout, sys.stderr
    try:
        captured = CapturedOutput()
        sys.stdout, sys.stderr = captured.stdout, captured.stderr
        yield captured
    finally:
        sys.stdout, sys.stderr = stdout, stderr
        captured.stdout = captured.stdout.getvalue()
        captured.stderr = captured.stderr.getvalue()

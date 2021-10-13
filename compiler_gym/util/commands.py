# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
from signal import Signals
from typing import List


def run_command(cmd: List[str], timeout: int):
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    stdout, stderr = communicate(process, timeout=timeout)
    if process.returncode:
        returncode = process.returncode
        try:
            # Try and decode the name of a signal. Signal returncodes
            # are negative.
            returncode = f"{returncode} ({Signals(abs(returncode)).name})"
        except ValueError:
            pass
        raise OSError(
            f"Compilation job failed with returncode {returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Stderr: {stderr.strip()}"
        )
    return stdout


def communicate(process, input=None, timeout=None):
    """subprocess.communicate() which kills subprocess on timeout."""
    try:
        return process.communicate(input=input, timeout=timeout)
    except subprocess.TimeoutExpired:
        # kill() was added in Python 3.7.
        if sys.version_info >= (3, 7, 0):
            process.kill()
        else:
            process.terminate()
        process.communicate(timeout=timeout)  # Wait for shutdown to complete.
        raise

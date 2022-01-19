# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
from contextlib import contextmanager
from signal import Signals
from subprocess import Popen as _Popen
from typing import List


def run_command(cmd: List[str], timeout: int):
    with Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    ) as process:
        stdout, stderr = process.communicate(timeout=timeout)
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
        # Wait for shutdown to complete.
        try:
            process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            pass  # Stubborn process won't die, nothing can be done.
        raise


@contextmanager
def Popen(*args, **kwargs):
    """subprocess.Popen() with resilient process termination at end of scope."""
    with _Popen(*args, **kwargs) as process:
        try:
            yield process
        finally:
            # Process has not yet terminated, kill it.
            if process.poll() is None:
                # kill() was added in Python 3.7.
                if sys.version_info >= (3, 7, 0):
                    process.kill()
                else:
                    process.terminate()
                # Wait for shutdown to complete.
                try:
                    process.communicate(timeout=60)
                except subprocess.TimeoutExpired:
                    pass  # Stubborn process won't die, nothing can be done.

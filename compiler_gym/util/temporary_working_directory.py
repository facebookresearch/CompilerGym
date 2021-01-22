# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A context manager to set a temporary working directory."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union


@contextmanager
def temporary_working_directory(directory: Optional[Union[str, Path]] = None) -> Path:
    """Temporarily set the working directory.

    :param directory: The directory to set as the temporary working directory.
    :return: The temporary working directory.
    """
    old_working_directory = os.getcwd()
    try:
        if directory:
            os.chdir(directory)
            yield Path(directory)
        else:
            with tempfile.TemporaryDirectory(prefix="compiler_gym-") as d:
                yield Path(d)
    finally:
        os.chdir(old_working_directory)

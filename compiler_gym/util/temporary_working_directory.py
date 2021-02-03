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

    This function provides a way to set the working directory within the
    scope of a "with statement". Example usage:

    .. code-block:: python

        print(os.getcwd())  # /tmp/foo
        with temporary_working_directory("/tmp/bar"):
            # Now in scope of new working directory.
            print(os.getcwd())  # /tmp/bar
        # Return to original working directory.
        print(os.getcwd())  # /tmp/foo

    :param directory: A directory to set as the temporary working directory. If
        not provided, a temporary directory is created and deleted once out of
        scope.
    :return: The temporary working directory.
    """
    old_working_directory = os.getcwd()
    try:
        if directory:
            os.chdir(directory)
            yield Path(directory)
        else:
            with tempfile.TemporaryDirectory(prefix="compiler_gym-") as d:
                os.chdir(d)
                yield Path(d)
    finally:
        os.chdir(old_working_directory)

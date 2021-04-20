# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for working with the filesystem."""
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO, List, TextIO, Union

from compiler_gym.util import runfiles_path


def get_storage_paths() -> List[Path]:
    """Return the list of paths used by CompilerGym for filesystem storage.

    :return: A list of filesystem paths that CompilerGym uses to store files.
    """
    return sorted(
        {
            runfiles_path.cache_path("."),
            runfiles_path.transient_cache_path("."),
            runfiles_path.site_data_path("."),
        }
    )


@contextmanager
def atomic_file_write(
    path: Path, fileobj: bool = False, mode: str = "wb"
) -> Union[Path, TextIO, BinaryIO]:
    """A context manager for atomically writing to a file.

    Provides a lock-free mechanism for ensuring concurrent safe writes to a
    filesystem path. Use this to prevent filesystem races when multiple callers
    may be writing to the same file. This is best suited for cases where the
    chance of a race are low, as it does not prevent redundant writes. It simply
    guarantees that each write is atomic.

    This relies on POSIX atomic file renaming.

    Use it as a context manager that yields the path of a temporary file to
    write to:

        >>> outpath = Path("some_file.txt")
        >>> with atomic_file_write(outpath) as tmp_path:
        ...     with open(tmp_path, "w") as f:
        ...         f.write("Hello\n")
        >>> outpath.is_file()
        True

    It can also return a file object if passed the :code:`fileobj` argument:

        >>> outpath = Path("some_file.txt")
        >>> with atomic_file_write(outpath, fileobj=True) as f:
        ...     f.write(file_data)
        >>> outpath.is_file()
        True

    :param path: The path to write to atomically write to.

    :param fileobj: If :code:`True`, return a file object in the given
        :code:`mode`.

    :param mode: The file mode to use when returning a file object.

    :returns: The path of a temporary file to write to.
    """
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False, mode=mode) as tmp:
        tmp_path = Path(tmp.name)
        try:
            yield tmp if fileobj else tmp_path
        finally:
            if tmp_path.is_file():
                os.rename(tmp_path, path)

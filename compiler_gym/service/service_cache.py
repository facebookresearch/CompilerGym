# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module defines a filesystem cache for services."""
import random
import shutil
from datetime import datetime
from pathlib import Path

from compiler_gym.util.runfiles_path import transient_cache_path

MAX_CACHE_CONFLICT_RETIRES: int = 1000


class ServiceCache:
    """A filesystem cache for use by managed services.

    This provides a directory in which a service can store temporary files and
    artifacts. A service can assume exclusive use of this cache. When supported,
    the cache will be in an in-memory filesystem. The cache contains two
    subdirectories: "logs", which may be used for storing log files, and "disk",
    which may be used for storing files that require being stored on a
    traditional filesystem. On some Linux distributions, in-memory filesystems
    do not permit executing files.
    """

    def __init__(self):
        for _ in range(MAX_CACHE_CONFLICT_RETIRES):
            random_hash = random.getrandbits(16)
            service_name = datetime.now().strftime(
                f"s/%m%dT%H%M%S-%f-{random_hash:04x}"
            )
            self.path: Path = transient_cache_path(service_name)
            # Guard against the unlikely scenario that there is a collision
            # between the randomly generated working directories of multiple
            # ServiceCache constructors.
            try:
                (self.path / "logs").mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                pass
        else:
            raise OSError(
                "Could not create a unique cache directory "
                f"after {MAX_CACHE_CONFLICT_RETIRES} retries."
            )

    def __truediv__(self, rhs) -> Path:
        """Supports 'cache / "path"' syntax."""
        return self.path / rhs

    def close(self):
        """Remove the cache directory. This must be called."""
        shutil.rmtree(self.path, ignore_errors=True)

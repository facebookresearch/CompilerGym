# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym/service/service_cache.py."""
from compiler_gym.service.service_cache import ServiceCache
from tests.test_main import main


def test_service_cache(cache: ServiceCache):
    cache = ServiceCache()
    try:
        assert cache.path.is_dir()
        assert (cache / "logs").is_dir()
    finally:
        cache.close()

    assert not cache.path.is_dir()


if __name__ == "__main__":
    main()

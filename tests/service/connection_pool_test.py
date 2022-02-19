# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for compiler_gym/service/connection_pool.py."""

import pytest

import compiler_gym
from compiler_gym.envs.llvm import LLVM_SERVICE_BINARY
from compiler_gym.service import ConnectionOpts, ServiceConnectionPool
from compiler_gym.errors import ServiceError
from tests.test_main import main

pytest_plugins = ["tests.pytest_plugins.llvm"]


@pytest.fixture(scope="function")
def pool() -> ServiceConnectionPool:
    with ServiceConnectionPool() as pool_:
        yield pool_


def test_service_pool_with_statement():
    with ServiceConnectionPool() as pool:
        assert not pool.closed
    assert pool.closed


def test_service_pool_double_close(pool: ServiceConnectionPool):
    assert not pool.closed
    pool.close()
    assert pool.closed
    pool.close()
    assert pool.closed


def test_service_pool_acquire_release(pool: ServiceConnectionPool):
    service = pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts())
    assert service in pool
    service.release()
    assert service in pool


def test_service_pool_contains(pool: ServiceConnectionPool):
    with ServiceConnectionPool() as other_pool:
        with pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts()) as service:
            assert service in pool
            assert service not in other_pool
            assert service not in ServiceConnectionPool.get()

        # Service remains in pool after release.
        assert service in pool


def test_service_pool_close_frees_service(pool: ServiceConnectionPool):
    service = pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts())
    assert not service.closed
    pool.close()
    assert service.closed


def test_service_pool_service_is_not_closed(pool: ServiceConnectionPool):
    service = None
    service = pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts())
    service.close()
    assert not service.closed


def test_service_pool_with_service_is_not_closed(pool: ServiceConnectionPool):
    service = None
    with pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts()) as service:
        assert not service.closed
    assert not service.closed


def test_service_pool_with_env_is_not_closed(pool: ServiceConnectionPool):
    with compiler_gym.make("llvm-v0", service_pool=pool) as env:
        service = env.service
        assert not service.closed
    assert not service.closed


def test_service_pool_fork(pool: ServiceConnectionPool):
    with compiler_gym.make("llvm-v0", service_pool=pool) as env:
        env.reset()
        with env.fork() as fkd:
            fkd.reset()
            assert env.service == fkd.service
            assert not env.service.closed
        assert not env.service.closed


def test_service_pool_release_service(pool: ServiceConnectionPool):
    service = pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts())
    service.close()
    # A released service remains alive.
    assert not service.closed


def test_service_pool_release_dead_service(pool: ServiceConnectionPool):
    service = pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts())
    service.shutdown()
    assert service.closed
    service.close()
    # A dead service cannot be reused, discard it.
    assert service not in pool


def test_service_pool_size(pool: ServiceConnectionPool):
    assert pool.size == 0
    assert len(pool) == pool.size

    with pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts()):
        assert pool.size == 1
        assert len(pool) == pool.size
        with pool.acquire(LLVM_SERVICE_BINARY, ConnectionOpts()):
            assert pool.size == 2
            assert len(pool) == pool.size


def test_service_pool_make_release(pool: ServiceConnectionPool):
    with compiler_gym.make("llvm-v0", service_pool=pool) as a:
        assert len(pool) == 1
        with compiler_gym.make("llvm-v0", service_pool=pool) as b:
            a_service = a.service
            b_service = b.service
            assert a_service != b_service
            assert len(pool) == 2

    with compiler_gym.make("llvm-v0", service_pool=pool) as c:
        c_service = c.service
        assert a_service == c_service
        assert a_service != b_service
        assert pool.size == 2


def test_service_pool_make_release_loop(pool: ServiceConnectionPool):
    for _ in range(5):
        with compiler_gym.make("llvm-v0", service_pool=pool):
            assert pool.size == 1
        assert pool.size == 1


def test_service_pool_environment_restarts_service(pool: ServiceConnectionPool):
    with compiler_gym.make("llvm-v0", service_pool=pool) as env:
        old_service = env.service
        env.service.shutdown()
        env.service.close()
        assert env.service.closed

        # For environment to restart service.
        env.reset()
        assert not env.service.closed

        new_service = env.service
        assert new_service in pool
        assert old_service not in pool


def test_service_pool_forked_service_dies(pool: ServiceConnectionPool):
    with compiler_gym.make("llvm-v0", service_pool=pool) as env:
        with env.fork() as fkd:
            assert env.service == fkd.service
            try:
                fkd.service.shutdown()
            except ServiceError:
                pass  # shutdown() raises service error if in-episode.
            fkd.service.close()

            env.reset()
            fkd.reset()
            assert env.service != fkd.service
            assert env.service in pool
            assert fkd.service in pool


# TODO: Test case where forked environment kills the service.

# TODO: Service pool connection does not interfere with pool.


if __name__ == "__main__":
    main()

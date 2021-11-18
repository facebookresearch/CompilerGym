# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from tempfile import TemporaryDirectory

from llvm_autotuning.optimization_target import OptimizationTarget

from compiler_gym.envs import CompilerEnv
from compiler_gym.random_search import random_search as lib_random_search
from compiler_gym.util.runfiles_path import transient_cache_path


def random(
    env: CompilerEnv,
    optimization_target: OptimizationTarget,
    search_time_seconds: int,
    patience: int = 350,
    **kwargs
) -> None:
    """Run a random search on the environment.

    :param env: The environment to optimize.

    :param optimization_target: The target to optimize for.

    :param search_time_seconds: The total search time.

    :param patience: The number of steps to search without an improvement before
        resetting to a new trajectory.
    """
    with TemporaryDirectory(
        dir=transient_cache_path("."), prefix="autotune-"
    ) as tmpdir:
        final_env = lib_random_search(
            make_env=lambda: optimization_target.make_env(env.benchmark),
            outdir=tmpdir,
            total_runtime=search_time_seconds,
            patience=patience,
            nproc=1,
        )
    env.apply(final_env.state)
    final_env.close()

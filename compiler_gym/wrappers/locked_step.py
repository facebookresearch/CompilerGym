# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from threading import Lock
from typing import Optional

from compiler_gym.envs import CompilerEnv
from compiler_gym.util.gym_type_hints import ObservationType, StepType
from compiler_gym.wrappers.core import CompilerEnvWrapper

_GLOBAL_STEP_LOCK = Lock()


class LockedStep(CompilerEnvWrapper):
    """A wrapper that protects environment operations with a thread lock.

    This class is used to prevent the :code:`step()`, :code:`reset()`, and
    :code:`fork()` methods of multiple compiler environments from executing
    simultaneously. It does this by sharing acquiring a thread lock during these
    operations which is shared between all instances of this wrapped class.

    Example usage:

        >>> envs = [compiler_gym.make("llvm-v0") for _ in range(10)]
        >>> threads = [threading.Thread(target=run, args=(env,)) for env in envs]
        >>> [t.start() for t in threads]
    """

    def __init__(self, env: CompilerEnv, lock: Optional[Lock] = None):
        """Constructor.

        :param env: The environment to wrap.

        :param lock: The thread lock to acquire when performing an operation.
            If not provided, a default lock is used and shared.
        """
        super().__init__(env)
        self.lock = lock or _GLOBAL_STEP_LOCK

    def reset(self, *args, **kwargs) -> ObservationType:
        with self.lock:
            return self.env.reset(*args, **kwargs)

    def step(self, *args, **kwargs) -> StepType:
        with self.lock:
            return self.env.step(*args, **kwargs)

    def fork(self):
        with self.lock:
            fkd = self.env.fork()
            return LockedStep(env=fkd, lock=self.lock)

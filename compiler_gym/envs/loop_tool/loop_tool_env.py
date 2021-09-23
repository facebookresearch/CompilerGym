# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.envs.compiler_env import CompilerEnv


class LoopToolEnv(CompilerEnv):
    def commandline(self):
        return ",".join(str(x) for x in self.actions)

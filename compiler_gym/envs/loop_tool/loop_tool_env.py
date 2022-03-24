# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from compiler_gym.service.client_service_compiler_env import ClientServiceCompilerEnv


class LoopToolEnv(ClientServiceCompilerEnv):
    def commandline(self):
        return ",".join(str(x) for x in self.actions)

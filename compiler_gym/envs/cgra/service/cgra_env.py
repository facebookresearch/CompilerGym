import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast
from compiler_gym.util.gym_type_hints import ObservationType, OptionalArgumentValue

import numpy as np
from compiler_gym.util.runfiles_path import site_data_path

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.envs.cgra.datasets import get_cgra_datasets

from compiler_gym.errors import BenchmarkInitError
from compiler_gym.service.client_service_compiler_env import ClientServiceCompilerEnv
from compiler_gym.spaces import Box, Commandline
from compiler_gym.spaces import Dict as DictSpace
from compiler_gym.spaces import Scalar, Sequence

from compiler_gym.envs.cgra.cgra_rewards import IntermediateIIReward

class CgraEnv(ClientServiceCompilerEnv):
    def __init__(self, *args, datasets_site_path: Optional[Path] = None, benchmark: Optional[Union[str, Benchmark]] = None, datasets_set_path: Optional[Path] = None, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            benchmark = benchmark or "dfg_10/1",
            datasets=get_cgra_datasets(site_data_base=datasets_site_path),
            rewards=[IntermediateIIReward()]
            ,
            derived_observation_spaces=[
                # {
                #     "id": "CurrentOperation",
                #     "base_id": 
                # }
            ]
        )

    def reset(self, reward_space = OptionalArgumentValue.UNCHANGED, *args, **kwargs):
        observation = super().reset(reward_space=reward_space, *args, **kwargs)

        return observation

    def make_benchmark(
        self, inputs, copt, system_include: bool = True, timeout: int=600
    ):
        # TOOD
        return None

    def render(self, mode="human"):
        if mode == "human":
            print("human-visible schedule")
        else:
            return self.render(mode)


from compiler_gym.util.gym_type_hints import OptionalArgumentValue
from compiler_gym.service.client_service_compiler_env import ClientServiceCompilerEnv
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast
from compiler_gym.datasets import Benchmark

from compiler_gym.envs.cgra.datasets import get_cgra_datasets
from compiler_gym.envs.cgra.cgra_rewards import IntermediateIIReward, FinalIIReward

class RelativeCgraEnv(ClientServiceCompilerEnv):
    def __init__(self, *args, punish_intermediate: bool = True, datasets_site_path: Optional[Path] = None, benchmark: Optional[Union[str, Benchmark]], **kwargs):
        if punish_intermediate:
            reward = IntermediateIIReward()
        else:
            reward = FinalIIReward()
        super().__init__(
            *args,
            **kwargs,
            benchmark = benchmark or "dfg_10/1",
            datasets=get_cgra_datasets(site_data_base=datasets_site_path),
            rewards=[reward],
            derived_observation_spaces=[]
        )

    def reset(self, reward_space = OptionalArgumentValue.UNCHANGED, *args, **kwargs):
        observation = super().reset(reward_space=reward_space, *args, **kwargs)

        return observation

    def make_benchmark(self, inputs, copt, system_include: bool = True, timeout: int=600):
        return None

    def render(self, mode="human"):
        if mode == "human":
            print("Human visible schedule")
        else:
            return self.render(mode)
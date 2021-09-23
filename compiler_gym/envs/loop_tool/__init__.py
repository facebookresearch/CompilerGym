# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Register the loop_tool environment and reward."""
from pathlib import Path
from typing import Iterable

from compiler_gym.datasets import Benchmark, Dataset, benchmark
from compiler_gym.spaces import Reward
from compiler_gym.util.registration import register
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

LOOP_TOOL_SERVICE_BINARY: Path = runfiles_path(
    "compiler_gym/envs/loop_tool/service/compiler_gym-loop_tool-service"
)


class FLOPSReward(Reward):
    """
    `loop_tool` uses "floating point operations per second"
    as its default reward space.
    """

    def __init__(self):
        super().__init__(
            id="flops",
            observation_spaces=["flops"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous_flops = None

    def reset(self, benchmark: str, observation_view):
        del benchmark  # unused
        self.previous_flops = observation_view["flops"]

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous_flops is None:
            self.previous_flops = observations[0]
            return self.previous_flops

        reward = float(observations[0] - self.previous_flops)
        self.previous_flops = observations[0]
        return reward


class LoopToolCUDADataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://loop_tool-cuda-v0",
            license="MIT",
            description="loop_tool dataset",
            site_data_base=site_data_path("loop_tool_dataset"),
        )

    def benchmark_uris(self) -> Iterable[str]:
        return (f"loop_tool-cuda-v0/{i}" for i in range(1, 1024 * 1024 * 8))

    def benchmark(self, uri: str) -> Benchmark:
        return Benchmark(proto=benchmark.BenchmarkProto(uri=uri))


class LoopToolCPUDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(
            name="benchmark://loop_tool-cpu-v0",
            license="MIT",
            description="loop_tool dataset",
            site_data_base=site_data_path("loop_tool_dataset"),
        )

    def benchmark_uris(self) -> Iterable[str]:
        return (f"loop_tool-cpu-v0/{i}" for i in range(1, 1024 * 1024 * 8))

    def benchmark(self, uri: str) -> Benchmark:
        return Benchmark(proto=benchmark.BenchmarkProto(uri=uri))


register(
    id="loop_tool-v0",
    entry_point="compiler_gym.envs.loop_tool.loop_tool_env:LoopToolEnv",
    kwargs={
        "datasets": [LoopToolCPUDataset(), LoopToolCUDADataset()],
        "observation_space": "action_state",
        "reward_space": "flops",
        "rewards": [FLOPSReward()],
        "service": LOOP_TOOL_SERVICE_BINARY,
    },
)

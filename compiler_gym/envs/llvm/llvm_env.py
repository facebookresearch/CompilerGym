# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Extensions to the CompilerEnv environment for LLVM."""
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast

import numpy as np
from gym.spaces import Dict as DictSpace

from compiler_gym.envs.compiler_env import CompilerEnv, step_t
from compiler_gym.envs.llvm.benchmarks import make_benchmark
from compiler_gym.envs.llvm.datasets import LLVM_DATASETS
from compiler_gym.envs.llvm.llvm_rewards import (
    BaselineImprovementNormalizedReward,
    CostFunctionReward,
    NormalizedReward,
)
from compiler_gym.spaces import Commandline, CommandlineFlag, Scalar, Sequence
from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_NAMES
from compiler_gym.third_party.inst2vec import Inst2vecEncoder
from compiler_gym.util.runfiles_path import runfiles_path, site_data_path

_ACTIONS_LIST = Path(
    runfiles_path("compiler_gym/envs/llvm/service/passes/actions_list.txt")
)

_FLAGS_LIST = Path(
    runfiles_path("compiler_gym/envs/llvm/service/passes/actions_flags.txt")
)

_DESCRIPTIONS_LIST = Path(
    runfiles_path("compiler_gym/envs/llvm/service/passes/actions_descriptions.txt")
)


def _read_list_file(path: Path) -> Iterable[str]:
    with open(str(path)) as f:
        for action in f:
            if action.strip():
                yield action.strip()


_ACTIONS = list(_read_list_file(_ACTIONS_LIST))
_FLAGS = dict(zip(_ACTIONS, _read_list_file(_FLAGS_LIST)))
_DESCRIPTIONS = dict(zip(_ACTIONS, _read_list_file(_DESCRIPTIONS_LIST)))
_INST2VEC_ENCODER = Inst2vecEncoder()


class LlvmEnv(CompilerEnv):
    """A specialized CompilerEnv for LLVM.

    This extends the default :class:`CompilerEnv` environment, adding extra LLVM
    functionality. Specifically, the actions use the
    :class:`CommandlineFlag <compiler_gym.spaces.CommandlineFlag>` space, which
    is a type of :code:`Discrete` space that provides additional documentation
    about each action, and the
    :meth:`LlvmEnv.commandline() <compiler_gym.envs.LlvmEnv.commandline>` method
    can be used to produce an equivalent LLVM opt invocation for the current
    environment state.

    :ivar actions: The list of actions that have been performed since the
        previous call to :func:`reset`.
    :vartype actions: List[int]
    """

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            rewards=[
                CostFunctionReward(
                    id="IrInstructionCount",
                    cost_function="IrInstructionCount",
                    init_cost_function="IrInstructionCountO0",
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                NormalizedReward(
                    id="IrInstructionCountNorm",
                    cost_function="IrInstructionCount",
                    init_cost_function="IrInstructionCountO0",
                    max=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                BaselineImprovementNormalizedReward(
                    id="IrInstructionCountO3",
                    cost_function="IrInstructionCount",
                    baseline_cost_function="IrInstructionCountO3",
                    init_cost_function="IrInstructionCountO0",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                BaselineImprovementNormalizedReward(
                    id="IrInstructionCountOz",
                    cost_function="IrInstructionCount",
                    baseline_cost_function="IrInstructionCountOz",
                    init_cost_function="IrInstructionCountO0",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                CostFunctionReward(
                    id="ObjectTextSizeBytes",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                NormalizedReward(
                    id="ObjectTextSizeNorm",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    max=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                BaselineImprovementNormalizedReward(
                    id="ObjectTextSizeO3",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    baseline_cost_function="ObjectTextSizeO3",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                BaselineImprovementNormalizedReward(
                    id="ObjectTextSizeOz",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    baseline_cost_function="ObjectTextSizeOz",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
            ],
        )
        self.actions: List[int] = []
        self.datasets_site_path = site_data_path("llvm/10.0.0/bitcode_benchmarks")

        # Register the LLVM datasets.
        self.datasets_site_path.mkdir(parents=True, exist_ok=True)
        self.inactive_datasets_site_path.mkdir(parents=True, exist_ok=True)
        for dataset in LLVM_DATASETS:
            self.register_dataset(dataset)

        self.inst2vec = _INST2VEC_ENCODER

        self.observation.spaces["CpuInfo"].space = DictSpace(
            {
                "name": Sequence(size_range=(0, None), dtype=str),
                "cores_count": Scalar(min=None, max=None, dtype=int),
                "l1i_cache_size": Scalar(min=None, max=None, dtype=int),
                "l1i_cache_count": Scalar(min=None, max=None, dtype=int),
                "l1d_cache_size": Scalar(min=None, max=None, dtype=int),
                "l1d_cache_count": Scalar(min=None, max=None, dtype=int),
                "l2_cache_size": Scalar(min=None, max=None, dtype=int),
                "l2_cache_count": Scalar(min=None, max=None, dtype=int),
                "l3_cache_size": Scalar(min=None, max=None, dtype=int),
                "l3_cache_count": Scalar(min=None, max=None, dtype=int),
                "l4_cache_size": Scalar(min=None, max=None, dtype=int),
                "l4_cache_count": Scalar(min=None, max=None, dtype=int),
            }
        )

        self.observation.add_derived_space(
            id="Inst2vecPreprocessedText",
            base_id="Ir",
            space=Sequence(size_range=(0, None), dtype=str),
            translate=lambda base_observation: self.inst2vec.preprocess(
                base_observation
            ),
            default_value="",
        )
        self.observation.add_derived_space(
            id="Inst2vecEmbeddingIndices",
            base_id="Ir",
            space=Sequence(size_range=(0, None), dtype=np.int32),
            translate=lambda base_observation: self.inst2vec.encode(
                self.inst2vec.preprocess(base_observation)
            ),
            default_value=np.array([self.inst2vec.vocab["!UNK"]]),
        )
        self.observation.add_derived_space(
            id="Inst2vec",
            base_id="Ir",
            space=Sequence(size_range=(0, None), dtype=np.ndarray),
            translate=lambda base_observation: self.inst2vec.embed(
                self.inst2vec.encode(self.inst2vec.preprocess(base_observation))
            ),
            default_value=np.vstack(
                [self.inst2vec.embeddings[self.inst2vec.vocab["!UNK"]]]
            ),
        )

        self.observation.add_derived_space(
            id="AutophaseDict",
            base_id="Autophase",
            space=DictSpace(
                {
                    name: Scalar(min=0, max=None, dtype=int)
                    for name in AUTOPHASE_FEATURE_NAMES
                }
            ),
            translate=lambda base_observation: {
                name: val
                for name, val in zip(AUTOPHASE_FEATURE_NAMES, base_observation)
            },
        )

    @staticmethod
    def make_benchmark(*args, **kwargs):
        """Alias to :func:`llvm.make_benchmark() <compiler_gym.envs.llvm.make_benchmark>`."""
        return make_benchmark(*args, **kwargs)

    def step(self, action: int) -> step_t:
        self.actions.append(action)
        return super().step(action)

    def reset(self, *args, **kwargs):
        self.actions = []
        return super().reset(*args, **kwargs)

    def _make_action_space(self, name: str, entries: List[str]) -> Commandline:
        flags = [
            CommandlineFlag(
                name=entry, flag=_FLAGS[entry], description=_DESCRIPTIONS[entry]
            )
            for entry in entries
        ]
        return Commandline(items=flags, name=name)

    def commandline(self) -> str:
        """Returns an LLVM :code:`opt` command line invocation for the current
        environment state.
        """
        command = cast(Commandline, self.action_space).commandline(self.actions)
        return f"opt {command} input.bc -o output.bc"

    def fork(self) -> "LlvmEnv":
        """Fork a new environment with exactly the same sate.

        This creates a duplicate environment instance with the current state.
        The new environment is entirely independently of the source
        episode and must be managed and
        :meth:`closed() <compiler_gym.envs.CompilerEnv.close>` by the user.

        Example usage:

        >>> env = gym.make("llvm-v0")
        # ... use env
        >>> new_env = env.fork()
        >>> new_env.actions == env.actions
        True

        :return: A new environment instance.
        """
        # Create a new environment using the same base settings as the current
        # environment.
        new_env = LlvmEnv(
            service=self._service_endpoint,
            observation_space=self.observation_space,
            reward_space=self.reward_space,
            action_space=self.action_space,
            connection_settings=self._connection_settings,
        )

        # Serialize the current program state to a bitcode file and use this to
        # initialize the state of the new environment.
        with tempfile.TemporaryDirectory(dir=self.service.connection.working_dir) as d:
            bitcode_file = Path(d) / "benchmark.bc"
            self.write_bitcode(bitcode_file)
            benchmark = new_env.make_benchmark(bitcode_file)
            new_env.reset(benchmark=benchmark)

            # This "custom benchmark" is only needed for initialization and
            # must be deleted. Otherwise calling new_env.fork() will try and
            # copy this file.
            del new_env._custom_benchmarks[benchmark.uri]

        # Copy over the mutable episode state.
        new_env.actions = self.actions.copy()
        new_env.episode_reward = self.episode_reward

        # Now that we have initialized the environment with the current state,
        # set the benchmark so that calls to new_env.reset() will correctly
        # revert the environment to the initial benchmark state.
        new_env._user_specified_benchmark_uri = self.benchmark
        # Set the "visible" name of the current benchmark to hide the fact that
        # we loaded from a custom bitcode file.
        new_env._benchmark_in_use_uri = self.benchmark

        # Re-register any custom benchmarks with the new environment.
        if self._custom_benchmarks:
            new_env._add_custom_benchmarks(
                list(self._custom_benchmarks.values()).copy()
            )

        return new_env

    @property
    def ir(self) -> str:
        """Print the LLVM-IR of the program in its current state.

        Alias for :code:`env.observation["Ir"]`.

        :return: A string of LLVM-IR.
        """
        return self.observation["Ir"]

    def write_bitcode(self, path: Union[Path, str]):
        """Write the current program state to a bitcode file.

        :param path: The path of the file to write.
        :return: The input :code:`path` argument.
        """
        path = Path(path).expanduser()
        tmp_path = self.observation["BitcodeFile"]
        try:
            shutil.copyfile(tmp_path, path)
        finally:
            os.unlink(tmp_path)

    def render(
        self,
        mode="human",
    ) -> Optional[str]:
        if mode == "human":
            print(self.ir)
        else:
            return super().render(mode)

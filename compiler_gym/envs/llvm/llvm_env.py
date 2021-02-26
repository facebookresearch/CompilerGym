# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Extensions to the CompilerEnv environment for LLVM."""
import hashlib
import os
import shutil
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Union, cast

import numpy as np
from gym.spaces import Dict as DictSpace

from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.envs.llvm.benchmarks import make_benchmark
from compiler_gym.envs.llvm.datasets import (
    LLVM_DATASETS,
    get_llvm_benchmark_validation_callback,
)
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
            translate=self.inst2vec.preprocess,
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

    def _make_action_space(self, name: str, entries: List[str]) -> Commandline:
        flags = [
            CommandlineFlag(
                name=entry, flag=_FLAGS[entry], description=_DESCRIPTIONS[entry]
            )
            for entry in entries
        ]
        return Commandline(items=flags, name=name)

    def commandline(  # pylint: disable=arguments-differ
        self, textformat: bool = False
    ) -> str:
        """Returns an LLVM :code:`opt` command line invocation for the current
        environment state.

        :param textformat: Whether to generate a command line that processes
            text-format LLVM-IR or bitcode (the default).
        :returns: A command line string.
        """
        command = cast(Commandline, self.action_space).commandline(self.actions)
        if textformat:
            return f"opt {command} input.ll -S -o output.ll"
        else:
            return f"opt {command} input.bc -o output.bc"

    def commandline_to_actions(self, commandline: str) -> List[int]:
        """Returns a list of actions from the given command line.

        :param commandline: A command line invocation, as generated by
            :meth:`env.commandline() <compiler_gym.envs.LlvmEnv.commandline>`.
        :return: A list of actions.
        :raises ValueError: In case the command line string is malformed.
        """
        # Strip the decorative elements that LlvmEnv.commandline() adds.
        if not commandline.startswith("opt "):
            raise ValueError(f"Invalid commandline: `{commandline}`")
        if commandline.endswith(" input.ll -S -o output.ll"):
            commandline = commandline[len("opt ") : -len(" input.ll -S -o output.ll")]
        elif commandline.endswith(" input.bc -o output.bc"):
            commandline = commandline[len("opt ") : -len(" input.bc -o output.bc")]
        else:
            raise ValueError(f"Invalid commandline: `{commandline}`")
        return self.action_space.from_commandline(commandline)

    @property
    def ir(self) -> str:
        """Print the LLVM-IR of the program in its current state.

        Alias for :code:`env.observation["Ir"]`.

        :return: A string of LLVM-IR.
        """
        return self.observation["Ir"]

    @property
    def ir_sha1(self) -> str:
        """Return the 40-characeter hex sha1 checksum of the current IR.

        Equivalent to: :code:`hashlib.sha1(env.ir.encode("utf-8")).hexdigest()`.

        :return: A 40-character hexademical sha1 string.
        """
        # TODO(cummins): Compute this on the service-side and add it as an
        # observation space.
        return hashlib.sha1(self.ir.encode("utf-8")).hexdigest()

    def write_ir(self, path: Union[Path, str]) -> Path:
        """Write the current program state to a file.

        :param path: The path of the file to write.
        :return: The input :code:`path` argument.
        """
        path = Path(path).expanduser()
        with open(path, "w") as f:
            f.write(self.ir)
        return path

    def write_bitcode(self, path: Union[Path, str]) -> Path:
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
        return path

    def render(
        self,
        mode="human",
    ) -> Optional[str]:
        if mode == "human":
            print(self.ir)
        else:
            return super().render(mode)

    def get_benchmark_validation_callback(
        self,
    ) -> Optional[Callable[[CompilerEnv], Optional[str]]]:
        """Return a callback for validating a given environment state.

        If there is no valid callback, returns :code:`None`.

        :param env: An :class:`LlvmEnv` instance.
        :return: An optional callback that takes an :class:`LlvmEnv` instance as
            argument and returns an optional string containing a validation error
            message.
        """
        return get_llvm_benchmark_validation_callback(self)

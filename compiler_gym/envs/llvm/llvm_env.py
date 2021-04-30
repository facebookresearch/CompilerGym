# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Extensions to the CompilerEnv environment for LLVM."""
import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Union, cast

import numpy as np
from gym.spaces import Box
from gym.spaces import Dict as DictSpace

from compiler_gym.datasets import Benchmark, BenchmarkInitError, Dataset
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.envs.llvm.datasets import get_llvm_datasets
from compiler_gym.envs.llvm.llvm_benchmark import ClangInvocation, make_benchmark
from compiler_gym.envs.llvm.llvm_rewards import (
    BaselineImprovementNormalizedReward,
    CostFunctionReward,
    NormalizedReward,
)
from compiler_gym.spaces import Commandline, CommandlineFlag, Scalar, Sequence
from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_NAMES
from compiler_gym.third_party.inst2vec import Inst2vecEncoder
from compiler_gym.third_party.llvm import download_llvm_files
from compiler_gym.third_party.llvm.instcount import INST_COUNT_FEATURE_NAMES
from compiler_gym.util.runfiles_path import runfiles_path

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


# TODO(github.com/facebookresearch/CompilerGym/issues/122): Replace text file
# parsing with build-generated python modules and import them.
_ACTIONS = list(_read_list_file(_ACTIONS_LIST))
_FLAGS = dict(zip(_ACTIONS, _read_list_file(_FLAGS_LIST)))
_DESCRIPTIONS = dict(zip(_ACTIONS, _read_list_file(_DESCRIPTIONS_LIST)))

_INST2VEC_ENCODER = Inst2vecEncoder()


_LLVM_DATASETS: Optional[List[Dataset]] = None


def _get_llvm_datasets(site_data_base: Optional[Path] = None) -> Iterable[Dataset]:
    """Get the LLVM datasets. Use a singleton value when site_data_base is the
    default value.
    """
    global _LLVM_DATASETS
    if site_data_base is None:
        if _LLVM_DATASETS is None:
            _LLVM_DATASETS = list(get_llvm_datasets(site_data_base=site_data_base))
        return _LLVM_DATASETS
    return get_llvm_datasets(site_data_base=site_data_base)


class LlvmEnv(CompilerEnv):
    """A specialized CompilerEnv for LLVM.

    This extends the default :class:`CompilerEnv
    <compiler_gym.envs.CompilerEnv>` environment, adding extra LLVM
    functionality. Specifically, the actions use the :class:`CommandlineFlag
    <compiler_gym.spaces.CommandlineFlag>` space, which is a type of
    :code:`Discrete` space that provides additional documentation about each
    action, and the :meth:`LlvmEnv.commandline()
    <compiler_gym.envs.LlvmEnv.commandline>` method can be used to produce an
    equivalent LLVM opt invocation for the current environment state.

    :ivar actions: The list of actions that have been performed since the
        previous call to :func:`reset`.

    :vartype actions: List[int]
    """

    def __init__(
        self,
        *args,
        benchmark: Optional[Union[str, Benchmark]] = None,
        datasets_site_path: Optional[Path] = None,
        **kwargs,
    ):
        # First perform a one-time download of LLVM binaries that are needed by
        # the LLVM service and are not included by the pip-installed package.
        download_llvm_files()
        super().__init__(
            *args,
            **kwargs,
            # Set a default benchmark for use.
            benchmark=benchmark or "cbench-v1/qsort",
            datasets=_get_llvm_datasets(site_data_base=datasets_site_path),
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
            id="InstCountDict",
            base_id="InstCount",
            space=DictSpace(
                {
                    f"{name}Count": Scalar(min=0, max=None, dtype=int)
                    for name in INST_COUNT_FEATURE_NAMES
                }
            ),
            translate=lambda base_observation: {
                f"{name}Count": val
                for name, val in zip(INST_COUNT_FEATURE_NAMES, base_observation)
            },
        )

        self.observation.add_derived_space(
            id="InstCountNorm",
            base_id="InstCount",
            space=Box(
                low=0,
                high=1,
                shape=(len(INST_COUNT_FEATURE_NAMES) - 1,),
                dtype=np.float32,
            ),
            translate=lambda base_observation: (
                base_observation[1:] / max(base_observation[0], 1)
            ).astype(np.float32),
        )

        self.observation.add_derived_space(
            id="InstCountNormDict",
            base_id="InstCountNorm",
            space=DictSpace(
                {
                    f"{name}Density": Scalar(min=0, max=None, dtype=int)
                    for name in INST_COUNT_FEATURE_NAMES[1:]
                }
            ),
            translate=lambda base_observation: {
                f"{name}Density": val
                for name, val in zip(INST_COUNT_FEATURE_NAMES[1:], base_observation)
            },
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

    def reset(self, *args, **kwargs):
        try:
            return super().reset(*args, **kwargs)
        except ValueError as e:
            # Catch and re-raise a compilation error with a more informative
            # error type.
            if "Failed to compute .text size cost" in str(e):
                raise BenchmarkInitError(
                    f"Failed to initialize benchmark {self._benchmark_in_use.uri}: {e}"
                ) from e
            raise

    def make_benchmark(
        self,
        inputs: Union[
            str, Path, ClangInvocation, List[Union[str, Path, ClangInvocation]]
        ],
        copt: Optional[List[str]] = None,
        system_includes: bool = True,
        timeout: int = 600,
    ) -> Benchmark:
        """Create a benchmark for use with this environment.

        This function takes one or more inputs and uses them to create a
        benchmark that can be passed to :meth:`compiler_gym.envs.LlvmEnv.reset`.

        For single-source C/C++ programs, you can pass the path of the source
        file:

            >>> benchmark = make_benchmark('my_app.c')
            >>> env = gym.make("llvm-v0")
            >>> env.reset(benchmark=benchmark)

        The clang invocation used is roughly equivalent to:

        .. code-block::

            $ clang my_app.c -O0 -c -emit-llvm -o benchmark.bc

        Additional compile-time arguments to clang can be provided using the
        :code:`copt` argument:

            >>> benchmark = make_benchmark('/path/to/my_app.cpp', copt=['-O2'])

        If you need more fine-grained control over the options, you can directly
        construct a :class:`ClangInvocation
        <compiler_gym.envs.llvm.ClangInvocation>` to pass a list of arguments to
        clang:

            >>> benchmark = make_benchmark(
                ClangInvocation(['/path/to/my_app.c'], timeout=10)
            )

        For multi-file programs, pass a list of inputs that will be compiled
        separately and then linked to a single module:

            >>> benchmark = make_benchmark([
                'main.c',
                'lib.cpp',
                'lib2.bc',
            ])

        If you already have prepared bitcode files, those can be linked and used
        directly:

            >>> benchmark = make_benchmark([
                'bitcode1.bc',
                'bitcode2.bc',
            ])

        .. note::

            LLVM bitcode compatibility is
            `not guaranteed <https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility>`_,
            so you must ensure that any precompiled bitcodes are compatible with the
            LLVM version used by CompilerGym, which can be queried using
            :func:`LlvmEnv.compiler_version <compiler_gym.envs.CompilerEnv.compiler_version>`.

        :param inputs: An input, or list of inputs.

        :param copt: A list of command line options to pass to clang when
            compiling source files.

        :param system_includes: Whether to include the system standard libraries
            during compilation jobs. This requires a system toolchain. See
            :func:`get_system_includes`.

        :param timeout: The maximum number of seconds to allow clang to run
            before terminating.

        :return: A :code:`Benchmark` instance.

        :raises FileNotFoundError: If any input sources are not found.

        :raises TypeError: If the inputs are of unsupported types.

        :raises OSError: If a compilation job fails.

        :raises TimeoutExpired: If a compilation job exceeds :code:`timeout`
            seconds.
        """
        return make_benchmark(
            inputs=inputs,
            copt=copt,
            system_includes=system_includes,
            timeout=timeout,
        )

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

        :return: A 40-character hexadecimal sha1 string.
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

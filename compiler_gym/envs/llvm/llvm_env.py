# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Extensions to the ClientServiceCompilerEnv environment for LLVM."""
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List, Optional, Union, cast

import numpy as np

from compiler_gym.datasets import Benchmark, Dataset
from compiler_gym.envs.llvm.benchmark_from_command_line import BenchmarkFromCommandLine
from compiler_gym.envs.llvm.datasets import get_llvm_datasets
from compiler_gym.envs.llvm.lexed_ir import LexedToken
from compiler_gym.envs.llvm.llvm_benchmark import (
    ClangInvocation,
    get_system_library_flags,
    make_benchmark,
)
from compiler_gym.envs.llvm.llvm_rewards import (
    BaselineImprovementNormalizedReward,
    CostFunctionReward,
    NormalizedReward,
)
from compiler_gym.errors import BenchmarkInitError, SessionNotFound
from compiler_gym.service.client_service_compiler_env import ClientServiceCompilerEnv
from compiler_gym.spaces import Box, Commandline
from compiler_gym.spaces import Dict as DictSpace
from compiler_gym.spaces import Scalar, Sequence
from compiler_gym.third_party.autophase import AUTOPHASE_FEATURE_NAMES
from compiler_gym.third_party.gccinvocation.gccinvocation import GccInvocation
from compiler_gym.third_party.inst2vec import Inst2vecEncoder
from compiler_gym.third_party.llvm import (
    clang_path,
    download_llvm_files,
    llvm_link_path,
)
from compiler_gym.third_party.llvm.instcount import INST_COUNT_FEATURE_NAMES
from compiler_gym.util.commands import Popen
from compiler_gym.util.runfiles_path import transient_cache_path
from compiler_gym.util.shell_format import join_cmd

_INST2VEC_ENCODER = Inst2vecEncoder()


_LLVM_DATASETS: Optional[List[Dataset]] = None

logger = logging.getLogger(__name__)


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


class LlvmEnv(ClientServiceCompilerEnv):
    """A specialized ClientServiceCompilerEnv for LLVM.

    This extends the default :class:`ClientServiceCompilerEnv
    <compiler_gym.envs.ClientServiceCompilerEnv>` environment, adding extra LLVM
    functionality. Specifically, the actions use the :class:`CommandlineFlag
    <compiler_gym.spaces.CommandlineFlag>` space, which is a type of
    :code:`Discrete` space that provides additional documentation about each
    action, and the :meth:`LlvmEnv.commandline()
    <compiler_gym.envs.LlvmEnv.commandline>` method can be used to produce an
    equivalent LLVM opt invocation for the current environment state.
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
        self.inst2vec = _INST2VEC_ENCODER
        super().__init__(
            *args,
            **kwargs,
            # Set a default benchmark for use.
            benchmark=benchmark or "cbench-v1/qsort",
            datasets=_get_llvm_datasets(site_data_base=datasets_site_path),
            rewards=[
                CostFunctionReward(
                    name="IrInstructionCount",
                    cost_function="IrInstructionCount",
                    init_cost_function="IrInstructionCountO0",
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                NormalizedReward(
                    name="IrInstructionCountNorm",
                    cost_function="IrInstructionCount",
                    init_cost_function="IrInstructionCountO0",
                    max=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                BaselineImprovementNormalizedReward(
                    name="IrInstructionCountO3",
                    cost_function="IrInstructionCount",
                    baseline_cost_function="IrInstructionCountO3",
                    init_cost_function="IrInstructionCountO0",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                BaselineImprovementNormalizedReward(
                    name="IrInstructionCountOz",
                    cost_function="IrInstructionCount",
                    baseline_cost_function="IrInstructionCountOz",
                    init_cost_function="IrInstructionCountO0",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=False,
                ),
                CostFunctionReward(
                    name="ObjectTextSizeBytes",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                NormalizedReward(
                    name="ObjectTextSizeNorm",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    max=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                BaselineImprovementNormalizedReward(
                    name="ObjectTextSizeO3",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    baseline_cost_function="ObjectTextSizeO3",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                BaselineImprovementNormalizedReward(
                    name="ObjectTextSizeOz",
                    cost_function="ObjectTextSizeBytes",
                    init_cost_function="ObjectTextSizeO0",
                    baseline_cost_function="ObjectTextSizeOz",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                CostFunctionReward(
                    name="TextSizeBytes",
                    cost_function="TextSizeBytes",
                    init_cost_function="TextSizeO0",
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                NormalizedReward(
                    name="TextSizeNorm",
                    cost_function="TextSizeBytes",
                    init_cost_function="TextSizeO0",
                    max=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                BaselineImprovementNormalizedReward(
                    name="TextSizeO3",
                    cost_function="TextSizeBytes",
                    init_cost_function="TextSizeO0",
                    baseline_cost_function="TextSizeO3",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
                BaselineImprovementNormalizedReward(
                    name="TextSizeOz",
                    cost_function="TextSizeBytes",
                    init_cost_function="TextSizeO0",
                    baseline_cost_function="TextSizeOz",
                    success_threshold=1,
                    default_negates_returns=True,
                    deterministic=True,
                    platform_dependent=True,
                ),
            ],
            derived_observation_spaces=[
                {
                    "id": "Inst2vecPreprocessedText",
                    "base_id": "Ir",
                    "space": Sequence(
                        name="Inst2vecPreprocessedText", size_range=(0, None), dtype=str
                    ),
                    "translate": self.inst2vec.preprocess,
                    "default_value": "",
                },
                {
                    "id": "Inst2vecEmbeddingIndices",
                    "base_id": "Ir",
                    "space": Sequence(
                        name="Inst2vecEmbeddingIndices",
                        size_range=(0, None),
                        dtype=np.int32,
                    ),
                    "translate": lambda base_observation: self.inst2vec.encode(
                        self.inst2vec.preprocess(base_observation)
                    ),
                    "default_value": np.array([self.inst2vec.vocab["!UNK"]]),
                },
                {
                    "id": "Inst2vec",
                    "base_id": "Ir",
                    "space": Sequence(
                        name="Inst2vec", size_range=(0, None), dtype=np.ndarray
                    ),
                    "translate": lambda base_observation: self.inst2vec.embed(
                        self.inst2vec.encode(self.inst2vec.preprocess(base_observation))
                    ),
                    "default_value": np.vstack(
                        [self.inst2vec.embeddings[self.inst2vec.vocab["!UNK"]]]
                    ),
                },
                {
                    "id": "InstCountDict",
                    "base_id": "InstCount",
                    "space": DictSpace(
                        {
                            f"{name}Count": Scalar(
                                name=f"{name}Count", min=0, max=None, dtype=int
                            )
                            for name in INST_COUNT_FEATURE_NAMES
                        },
                        name="InstCountDict",
                    ),
                    "translate": lambda base_observation: {
                        f"{name}Count": val
                        for name, val in zip(INST_COUNT_FEATURE_NAMES, base_observation)
                    },
                },
                {
                    "id": "InstCountNorm",
                    "base_id": "InstCount",
                    "space": Box(
                        name="InstCountNorm",
                        low=0,
                        high=1,
                        shape=(len(INST_COUNT_FEATURE_NAMES) - 1,),
                        dtype=np.float32,
                    ),
                    "translate": lambda base_observation: (
                        base_observation[1:] / max(base_observation[0], 1)
                    ).astype(np.float32),
                },
                {
                    "id": "InstCountNormDict",
                    "base_id": "InstCountNorm",
                    "space": DictSpace(
                        {
                            f"{name}Density": Scalar(
                                name=f"{name}Density", min=0, max=None, dtype=int
                            )
                            for name in INST_COUNT_FEATURE_NAMES[1:]
                        },
                        name="InstCountNormDict",
                    ),
                    "translate": lambda base_observation: {
                        f"{name}Density": val
                        for name, val in zip(
                            INST_COUNT_FEATURE_NAMES[1:], base_observation
                        )
                    },
                },
                {
                    "id": "AutophaseDict",
                    "base_id": "Autophase",
                    "space": DictSpace(
                        {
                            name: Scalar(name=name, min=0, max=None, dtype=int)
                            for name in AUTOPHASE_FEATURE_NAMES
                        },
                        name="AutophaseDict",
                    ),
                    "translate": lambda base_observation: {
                        name: val
                        for name, val in zip(AUTOPHASE_FEATURE_NAMES, base_observation)
                    },
                },
                {
                    "id": "LexedIrTuple",
                    "base_id": "LexedIr",
                    "space": Sequence(
                        name="LexedToken",
                        size_range=(0, None),
                        dtype=LexedToken,
                    ),
                    "translate": lambda base_observation: [
                        LexedToken(tid, kind, cat, val)
                        for tid, kind, cat, val in zip(
                            base_observation["token_id"],
                            base_observation["token_kind"],
                            base_observation["token_category"],
                            base_observation["token_value"],
                        )
                    ],
                    "default_value": {
                        "token_id": [],
                        "token_kind": [],
                        "token_category": [],
                        "token_value": [],
                    },
                },
            ],
        )

        # Mutable runtime configuration options that must be set on every call
        # to reset.
        self._runtimes_per_observation_count: Optional[int] = None
        self._runtimes_warmup_per_observation_count: Optional[int] = None

        cpu_info_spaces = [
            Sequence(name="name", size_range=(0, None), dtype=str),
            Scalar(name="cores_count", min=None, max=None, dtype=int),
            Scalar(name="l1i_cache_size", min=None, max=None, dtype=int),
            Scalar(name="l1i_cache_count", min=None, max=None, dtype=int),
            Scalar(name="l1d_cache_size", min=None, max=None, dtype=int),
            Scalar(name="l1d_cache_count", min=None, max=None, dtype=int),
            Scalar(name="l2_cache_size", min=None, max=None, dtype=int),
            Scalar(name="l2_cache_count", min=None, max=None, dtype=int),
            Scalar(name="l3_cache_size", min=None, max=None, dtype=int),
            Scalar(name="l3_cache_count", min=None, max=None, dtype=int),
            Scalar(name="l4_cache_size", min=None, max=None, dtype=int),
            Scalar(name="l4_cache_count", min=None, max=None, dtype=int),
        ]
        self.observation.spaces["CpuInfo"].space = DictSpace(
            {space.name: space for space in cpu_info_spaces},
            name="CpuInfo",
        )

    def reset(self, *args, **kwargs):
        try:
            return super().reset(*args, **kwargs)
        except ValueError as e:
            # Catch and re-raise some known benchmark initialization errors with
            # a more informative error type.
            if "Failed to compute .text size cost" in str(e):
                raise BenchmarkInitError(
                    f"Failed to initialize benchmark {self._benchmark_in_use.uri}: {e}"
                ) from e
            elif (
                "File not found:" in str(e)
                or "File is empty:" in str(e)
                or "Error reading file:" in str(e)
            ):
                raise BenchmarkInitError(str(e)) from e
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

        This function takes one or more inputs and uses them to create an LLVM
        bitcode benchmark that can be passed to
        :meth:`compiler_gym.envs.LlvmEnv.reset`.

        The following input types are supported:

        +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
        | **File Suffix**                                     | **Treated as**      | **Converted using**                                         |
        +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
        | :code:`.bc`                                         | LLVM IR bitcode     | No conversion required.                                     |
        +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
        | :code:`.ll`                                         | LLVM IR text format | Assembled to bitcode using llvm-as.                         |
        +-----------------------------------------------------+---------------------+-------------------------------------------------------------+
        | :code:`.c`, :code:`.cc`, :code:`.cpp`, :code:`.cxx` | C / C++ source      | Compiled to bitcode using clang and the given :code:`copt`. |
        +-----------------------------------------------------+---------------------+-------------------------------------------------------------+

        .. note::

            The LLVM IR format has no compatability guarantees between versions (see
            `LLVM docs
            <https://llvm.org/docs/DeveloperPolicy.html#ir-backwards-compatibility>`_).
            You must ensure that any :code:`.bc` and :code:`.ll` files are
            compatible with the LLVM version used by CompilerGym, which can be
            reported using :func:`env.compiler_version
            <compiler_gym.envs.ClientServiceCompilerEnv.compiler_version>`.

        E.g. for single-source C/C++ programs, you can pass the path of the source
        file:

            >>> benchmark = env.make_benchmark('my_app.c')
            >>> env = gym.make("llvm-v0")
            >>> env.reset(benchmark=benchmark)

        The clang invocation used is roughly equivalent to:

        .. code-block::

            $ clang my_app.c -O0 -c -emit-llvm -o benchmark.bc

        Additional compile-time arguments to clang can be provided using the
        :code:`copt` argument:

            >>> benchmark = env.make_benchmark('/path/to/my_app.cpp', copt=['-O2'])

        If you need more fine-grained control over the options, you can directly
        construct a :class:`ClangInvocation
        <compiler_gym.envs.llvm.ClangInvocation>` to pass a list of arguments to
        clang:

            >>> benchmark = env.make_benchmark(
                ClangInvocation(['/path/to/my_app.c'], system_includes=False, timeout=10)
            )

        For multi-file programs, pass a list of inputs that will be compiled
        separately and then linked to a single module:

            >>> benchmark = env.make_benchmark([
                'main.c',
                'lib.cpp',
                'lib2.bc',
                'foo/input.bc'
            ])

        :param inputs: An input, or list of inputs.

        :param copt: A list of command line options to pass to clang when
            compiling source files.

        :param system_includes: Whether to include the system standard libraries
            during compilation jobs. This requires a system toolchain. See
            :func:`get_system_library_flags`.

        :param timeout: The maximum number of seconds to allow clang to run
            before terminating.

        :return: A :code:`Benchmark` instance.

        :raises FileNotFoundError: If any input sources are not found.

        :raises TypeError: If the inputs are of unsupported types.

        :raises OSError: If a suitable compiler cannot be found.

        :raises BenchmarkInitError: If a compilation job fails.

        :raises TimeoutExpired: If a compilation job exceeds :code:`timeout`
            seconds.
        """
        return make_benchmark(
            inputs=inputs,
            copt=copt,
            system_includes=system_includes,
            timeout=timeout,
        )

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
        return self.observation["IrSha1"]

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

    @property
    def runtime_observation_count(self) -> int:
        """The number of runtimes to return for the Runtime observation space.

        See the :ref:`Runtime observation space reference <llvm/index:Runtime>`
        for further details.

        Example usage:

            >>> env = compiler_gym.make("llvm-v0")
            >>> env.reset()
            >>> env.runtime_observation_count = 10
            >>> len(env.observation.Runtime())
            10

        :getter: Returns the number of runtimes that will be returned when a
            :code:`Runtime` observation is requested.

        :setter: Set the number of runtimes to compute when a :code:`Runtime`
            observation is requested.

        :type: int
        """
        return self._runtimes_per_observation_count or int(
            self.send_param("llvm.get_runtimes_per_observation_count", "")
        )

    @runtime_observation_count.setter
    def runtime_observation_count(self, n: int) -> None:
        try:
            self.send_param(
                "llvm.set_runtimes_per_observation_count", str(n), resend_on_reset=True
            )
        except SessionNotFound:
            pass  # Not in session yet, will be sent on reset().
        self._runtimes_per_observation_count = n

    @property
    def runtime_warmup_runs_count(self) -> int:
        """The number of warmup runs of the binary to perform before measuring
        the Runtime observation space.

        See the :ref:`Runtime observation space reference <llvm/index:Runtime>`
        for further details.

        Example usage:

            >>> env = compiler_gym.make("llvm-v0")
            >>> env.reset()
            >>> env.runtime_observation_count = 10
            >>> len(env.observation.Runtime())
            10

        :getter: Returns the number of runs that be performed before measuring
            the :code:`Runtime` observation is requested.

        :setter: Set the number of warmup runs to perform when a :code:`Runtime`
            observation is requested.

        :type: int
        """
        return self._runtimes_warmup_per_observation_count or int(
            self.send_param("llvm.get_warmup_runs_count_per_runtime_observation", "")
        )

    @runtime_warmup_runs_count.setter
    def runtime_warmup_runs_count(self, n: int) -> None:
        try:
            self.send_param(
                "llvm.set_warmup_runs_count_per_runtime_observation",
                str(n),
                resend_on_reset=True,
            )
        except SessionNotFound:
            pass  # Not in session yet, will be sent on reset().
        self._runtimes_warmup_per_observation_count = n

    def fork(self):
        fkd = super().fork()
        if self.runtime_observation_count is not None:
            fkd.runtime_observation_count = self.runtime_observation_count
        if self.runtime_warmup_runs_count is not None:
            fkd.runtime_warmup_runs_count = self.runtime_warmup_runs_count
        return fkd

    def make_benchmark_from_command_line(
        self,
        cmd: Union[str, List[str]],
        replace_driver: bool = True,
        system_includes: bool = True,
        timeout: int = 600,
    ) -> Benchmark:
        """Create a benchmark for use with this environment.

        This function takes a command line compiler invocation as input,
        modifies it to produce an unoptimized LLVM-IR bitcode, and then runs the
        modified command line to produce a bitcode benchmark.

        For example, the command line:

            >>> benchmark = env.make_benchmark_from_command_line(
            ...     ["gcc", "-DNDEBUG", "a.c", "b.c", "-o", "foo", "-lm"]
            ... )

        Will compile a.c and b.c to an unoptimized benchmark that can be then
        passed to :meth:`reset() <compiler_env.envs.CompilerEnv.reset>`.

        The way this works is to change the first argument of the command line
        invocation to the version of clang shipped with CompilerGym, and to then
        append command line flags that causes the compiler to produce LLVM-IR
        with optimizations disabled. For example the input command line:

        .. code-block::

            gcc -DNDEBUG a.c b.c -o foo -lm

        Will be rewritten to be roughly equivalent to:

        .. code-block::

            /path/to/compiler_gym/clang -DNDEG a.c b.c \\
                -Xclang -disable-llvm-passes -Xclang -disable-llvm-optzns \\ -c
                -emit-llvm  -o -

        The generated benchmark then has a method :meth:`compile()
        <compiler_env.envs.llvm.BenchmarkFromCommandLine.compile>` which
        completes the linking and compilatilion to executable. For the above
        example, this would be roughly equivalent to:

        .. code-block::

            /path/to/compiler_gym/clang environment-bitcode.bc -o foo -lm

        :param cmd: A command line compiler invocation, either as a list of
            arguments (e.g. :code:`["clang", "in.c"]`) or as a single shell
            string (e.g. :code:`"clang in.c"`).

        :param replace_driver: Whether to replace the first argument of the
            command with the clang driver used by this environment.

        :param system_includes: Whether to include the system standard libraries
            during compilation jobs. This requires a system toolchain. See
            :func:`get_system_library_flags`.

        :param timeout: The maximum number of seconds to allow the compilation
            job to run before terminating.

        :return: A :class:`BenchmarkFromCommandLine
            <compiler_gym.envs.llvm.BenchmarkFromCommandLine>` instance.

        :raises ValueError: If no command line is provided.

        :raises BenchmarkInitError: If executing the command line fails.

        :raises TimeoutExpired: If a compilation job exceeds :code:`timeout`
            seconds.
        """
        if not cmd:
            raise ValueError("Input command line is empty")

        # Split the command line if passed a single string.
        if isinstance(cmd, str):
            cmd = shlex.split(cmd)

        rewritten_cmd: List[str] = cmd.copy()

        if len(cmd) < 2:
            raise ValueError(f"Input command line '{join_cmd(cmd)}' is too short")

        # Append include flags for the system headers if requested.
        if system_includes:
            rewritten_cmd += get_system_library_flags()

        # Use the CompilerGym clang binary in place of the original driver.
        if replace_driver:
            rewritten_cmd[0] = str(clang_path())

        # Strip the -S flag, if present, as that changes the output format.
        rewritten_cmd = [c for c in rewritten_cmd if c != "-S"]

        invocation = GccInvocation(rewritten_cmd)

        # Strip the output specifier(s). This is not strictly required since we
        # override it later, but makes the generated command easier to
        # understand.
        for i in range(len(rewritten_cmd) - 2, -1, -1):
            if rewritten_cmd[i] == "-o":
                del rewritten_cmd[i + 1]
                del rewritten_cmd[i]

        # Fail early.
        if "-" in invocation.sources:
            raise ValueError(
                "Input command line reads from stdin, "
                f"which is not supported: '{join_cmd(cmd)}'"
            )

        # Convert all of the C/C++ sources to bitcodes which can then be linked
        # into a single bitcode. We must process them individually because the
        # '-c' flag does not support multiple sources when we are specifying the
        # output path using '-o'.
        sources = set(s for s in invocation.sources if not s.endswith(".o"))

        if not sources:
            raise ValueError(
                f"Input command line has no source file inputs: '{join_cmd(cmd)}'"
            )

        bitcodes: List[bytes] = []
        for source in sources:
            # Adapt and execute the command line so that it will generate an
            # unoptimized bitecode file.
            emit_bitcode_command = rewritten_cmd.copy()

            # Strip the name of other sources:
            if len(sources) > 1:
                emit_bitcode_command = [
                    c for c in emit_bitcode_command if c == source or c not in sources
                ]

            # Append the flags to emit the bitcode and disable the optimization
            # passes.
            emit_bitcode_command += [
                "-c",
                "-emit-llvm",
                "-o",
                "-",
                "-Xclang",
                "-disable-llvm-passes",
                "-Xclang",
                "-disable-llvm-optzns",
            ]

            with Popen(
                emit_bitcode_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            ) as clang:
                logger.debug(
                    f"Generating LLVM bitcode benchmark: {join_cmd(emit_bitcode_command)}"
                )
                bitcode, stderr = clang.communicate(timeout=timeout)
                if clang.returncode:
                    raise BenchmarkInitError(
                        f"Failed to generate LLVM bitcode with error:\n"
                        f"{stderr.decode('utf-8').rstrip()}\n"
                        f"Running command: {join_cmd(emit_bitcode_command)}\n"
                        f"From original commandline: {join_cmd(cmd)}"
                    )
                bitcodes.append(bitcode)

        # If there were multiple sources then link the bitcodes together.
        if len(bitcodes) > 1:
            with TemporaryDirectory(
                dir=transient_cache_path("."), prefix="llvm-benchmark-"
            ) as dir:
                # Write the bitcodes to files.
                for i, bitcode in enumerate(bitcodes):
                    with open(os.path.join(dir, f"{i}.bc"), "wb") as f:
                        f.write(bitcode)

                # Link the bitcode files.
                llvm_link_cmd = [str(llvm_link_path()), "-o", "-"] + [
                    os.path.join(dir, f"{i}.bc") for i in range(len(bitcodes))
                ]
                with Popen(
                    llvm_link_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ) as llvm_link:
                    bitcode, stderr = llvm_link.communicate(timeout=timeout)
                    if llvm_link.returncode:
                        raise BenchmarkInitError(
                            f"Failed to link LLVM bitcodes with error: {stderr.decode('utf-8')}"
                        )

        return BenchmarkFromCommandLine(invocation, bitcode, timeout)

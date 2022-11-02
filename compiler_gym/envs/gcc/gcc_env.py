# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A CompilerGym environment for GCC."""
import codecs
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from compiler_gym.datasets import Benchmark
from compiler_gym.envs.gcc.datasets import get_gcc_datasets
from compiler_gym.envs.gcc.gcc import Gcc, GccSpec
from compiler_gym.envs.gcc.gcc_rewards import AsmSizeReward, ObjSizeReward
from compiler_gym.service import ConnectionOpts
from compiler_gym.service.client_service_compiler_env import ClientServiceCompilerEnv
from compiler_gym.spaces import Reward
from compiler_gym.util.decorators import memoized_property
from compiler_gym.util.gym_type_hints import ObservationType, OptionalArgumentValue
from compiler_gym.views import ObservationSpaceSpec

# The default gcc_bin argument.
DEFAULT_GCC: str = "docker:gcc:11.2.0"


class GccEnv(ClientServiceCompilerEnv):
    """A specialized ClientServiceCompilerEnv for GCC.

    This class exposes the optimization space of GCC's command line flags
    as an environment for reinforcement learning. For further details, see the
    :ref:`GCC Environment Reference <envs/gcc:Installation>`.
    """

    def __init__(
        self,
        *args,
        gcc_bin: Union[str, Path] = DEFAULT_GCC,
        benchmark: Union[str, Benchmark] = "benchmark://chstone-v0/adpcm",
        datasets_site_path: Optional[Path] = None,
        connection_settings: Optional[ConnectionOpts] = None,
        **kwargs,
    ):
        """Create an environment.

        :param gcc_bin: The path to the GCC executable, or the name of a docker
            image to use if prefixed with :code:`docker:`. Only used if the
            environment is attached to a local service. If attached remotely,
            the service will have already been created.

        :param benchmark: The benchmark to use for this environment. Either a
            URI string, or a :class:`Benchmark
            <compiler_gym.datasets.Benchmark>` instance. If not provided, a
            default benchmark is used.

        :param connection_settings: The connection settings to use.

        :raises EnvironmentNotSupported: If the runtime requirements for the GCC
            environment have not been met.

        :raises ServiceInitError: If the requested GCC version cannot be used.
        """
        connection_settings = connection_settings or ConnectionOpts()
        # Pass the executable path via an environment variable
        connection_settings.script_env = {"CC": gcc_bin}

        # Eagerly create a GCC compiler instance now because:
        #
        # 1. We want to catch an invalid gcc_bin argument early.
        #
        # 2. We want to perform the expensive one-off `docker pull` before we
        #    start the backend service, as otherwise the backend service
        #    initialization may time out.
        Gcc(bin=gcc_bin)

        super().__init__(
            *args,
            **kwargs,
            benchmark=benchmark,
            datasets=get_gcc_datasets(
                gcc_bin=gcc_bin, site_data_base=datasets_site_path
            ),
            rewards=[AsmSizeReward(), ObjSizeReward()],
            connection_settings=connection_settings,
        )

    def reset(
        self,
        benchmark: Optional[Union[str, Benchmark]] = None,
        action_space: Optional[str] = None,
        observation_space: Union[
            OptionalArgumentValue, str, ObservationSpaceSpec
        ] = OptionalArgumentValue.UNCHANGED,
        reward_space: Union[
            OptionalArgumentValue, str, Reward
        ] = OptionalArgumentValue.UNCHANGED,
    ) -> Optional[ObservationType]:
        observation = super().reset(
            benchmark=benchmark,
            action_space=action_space,
            observation_space=observation_space,
            reward_space=reward_space,
        )
        return observation

    def commandline(self) -> str:
        """Return a string representing the command line options.

        :return: A string.
        """
        return self.observation["command_line"]

    @memoized_property
    def gcc_spec(self) -> GccSpec:
        """A :class:`GccSpec <compiler_gym.envs.gcc.gcc.GccSpec>` description of
        the compiler specification.
        """
        pickled = self.send_param("gcc_spec", "")
        return pickle.loads(codecs.decode(pickled.encode(), "base64"))

    @property
    def source(self) -> str:
        """Get the source code."""
        return self.observation["source"]

    @property
    def rtl(self) -> str:
        """Get the final rtl of the program."""
        return self.observation["rtl"]

    @property
    def asm(self) -> str:
        """Get the assembly code."""
        return self.observation["asm"]

    @property
    def asm_size(self) -> int:
        """Get the assembly code size in bytes."""
        return self.observation["asm_size"]

    @property
    def asm_hash(self) -> str:
        """Get a hash of the assembly code."""
        return self.observation["asm_hash"]

    @property
    def instruction_counts(self) -> Dict[str, int]:
        """Get a count of the instruction types in the assembly code.

        Note, that it will also count fields beginning with a :code:`.`, like
        :code:`.bss` and :code:`.align`. Make sure to remove those if not
        needed.
        """
        return json.loads(self.observation["instruction_counts"])

    @property
    def obj(self) -> bytes:
        """Get the object code."""
        return self.observation["obj"]

    @property
    def obj_size(self) -> int:
        """Get the object code size in bytes."""
        return self.observation["obj_size"]

    @property
    def obj_hash(self) -> str:
        """Get a hash of the object code."""
        return self.observation["obj_hash"]

    @property
    def choices(self) -> List[int]:
        """Get the current choices"""
        return self.observation["choices"]

    @choices.setter
    def choices(self, choices: List[int]):
        """Set the current choices.

        This must be a list of ints with one element for each option the
        gcc_spec. Each element must be in range for the corresponding option.
        I.e. it must be between -1 and len(option) inclusive.
        """
        # TODO(github.com/facebookresearch/CompilerGym/issues/52): This can be
        # exposed directly through the action space once #369 is merged.
        assert len(self.gcc_spec.options) == len(choices)
        assert all(
            -1 <= c < len(self.gcc_spec.options[i]) for i, c in enumerate(choices)
        )
        self.send_param("choices", ",".join(map(str, choices)))

    def _init_kwargs(self) -> Dict[str, Any]:
        """Return the arguments required to initialize a GccEnv."""
        return {
            # GCC has an additional gcc_bin argument.
            "gcc_bin": self.gcc_spec.gcc.bin,
            **super()._init_kwargs(),
        }

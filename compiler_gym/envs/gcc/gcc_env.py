# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module demonstrates how to """
import codecs
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Union

from compiler_gym.datasets import Benchmark
from compiler_gym.envs.compiler_env import CompilerEnv
from compiler_gym.envs.gcc.datasets import get_gcc_datasets
from compiler_gym.envs.gcc.gcc import Gcc, GccSpec
from compiler_gym.service import ConnectionOpts
from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import ObservationType
from compiler_gym.views.observation import ObservationView

# The default gcc_bin argument.
DEFAULT_GCC: str = "docker:gcc:11.2.0"


class AsmSizeReward(Reward):
    """Reward for the size in bytes of the assembly code"""

    def __init__(self):
        super().__init__(
            id="asm_size",
            observation_spaces=["asm_size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous = None

    def reset(self, benchmark: str, observation_view: ObservationView):
        super().reset(benchmark, observation_view)
        del benchmark  # unused
        self.previous = None

    def update(self, action, observations, observation_view):
        del action  # unused
        del observation_view  # unused

        if self.previous is None:
            self.previous = observations[0]

        reward = float(self.previous - observations[0])
        self.previous = observations[0]
        return reward


class ObjSizeReward(Reward):
    """Reward for the size in bytes of the object code"""

    def __init__(self):
        super().__init__(
            id="obj_size",
            observation_spaces=["obj_size"],
            default_value=0,
            default_negates_returns=True,
            deterministic=False,
            platform_dependent=True,
        )
        self.previous = None

    def reset(self, benchmark: str, observation_view: ObservationView):
        super().reset(benchmark, observation_view)
        del benchmark  # unused
        self.previous = None

    def update(self, action, observations, observation_view):
        del action
        del observation_view

        if self.previous is None:
            self.previous = observations[0]

        reward = float(self.previous - observations[0])
        self.previous = observations[0]
        return reward


class GccEnv(CompilerEnv):
    """A compiler environment for GCC"""

    def __init__(
        self,
        *args,
        benchmark: Optional[Union[str, Benchmark]] = None,
        datasets_site_path: Optional[Path] = None,
        gcc_bin: Union[str, Path] = DEFAULT_GCC,
        connection_settings: Optional[ConnectionOpts] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        """Create an environment.

        :param gcc_bin: The path to the GCC executable. Only used if the
            environment is attached to a local service. If attached remotely,
            the service will have already been created.

        :param connection_settings: The connection settings to use.

        :param timeout: The timeout to use when compiling.
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
            # Set a default benchmark for use.
            benchmark=benchmark or "chstone-v0/adpcm",
            datasets=list(get_gcc_datasets(site_data_base=datasets_site_path)),
            rewards=[AsmSizeReward(), ObjSizeReward()],
            connection_settings=connection_settings,
        )
        self._spec = None
        self._timeout = timeout

    def reset(
        self,
        benchmark: Optional[Union[str, Benchmark]] = None,
        action_space: Optional[str] = None,
        retry_count: int = 0,
    ) -> Optional[ObservationType]:
        """Reset the environment. This additionally sets the timeout to the
        correct value."""
        observation = super().reset(benchmark, action_space, retry_count)
        if self._timeout:
            self.send_param("timeout", str(self._timeout))
        return observation

    @property
    def timeout(self) -> Optional[int]:
        """Get the current compilation timeout"""
        return self._timeout

    @timeout.setter
    def timeout(self, value: Optional[int]):
        """Tell the service what the compilation timeout is."""
        self._timeout = value
        self.send_param("timeout", str(value) if value else "")

    @property
    def gcc_spec(self) -> GccSpec:
        """Get a description of the GCC specification"""
        if not self._spec:
            pickled = self.send_param("gcc_spec", "")
            self._spec = pickle.loads(codecs.decode(pickled.encode(), "base64"))
        return self._spec

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
        Note, that it will also count fields beginning with a '.', like '.bss'
        and '.align'. Make sure to remove those if not needed."""
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
    def command_line(self) -> str:
        """Get the command line to compile the source."""
        return self.observation["command_line"]

    @property
    def choices(self) -> List[int]:
        """Get the current choices"""
        return self.observation["choices"]

    @choices.setter
    def choices(self, choices: List[int]):
        """Set the current choices.
        This must be a list of ints with one element for each option the
        gcc_spec.
        Each element must be in range for the corresponding option. I.e. it must
        be between -1 and len(option) inclusive."""
        spec = self.gcc_spec
        assert len(spec.options) == len(choices)
        assert all(-1 <= c < len(spec.options[i]) for i, c in enumerate(choices))
        self.send_param("choices", ",".join(map(str, choices)))

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
from typing import List, Optional, Tuple

from compiler_gym.service.proto import ActionSpace, Benchmark
from compiler_gym.service.proto import Event as Action
from compiler_gym.service.proto import Event as Observation
from compiler_gym.service.proto import ObservationSpace


class CompilationSession:
    """Base class for encapsulating an incremental compilation session.

    To add support for a new compiler, subclass from this base and provide
    implementations of the abstract methods, then call
    :func:`create_and_run_compiler_service
    <compiler_gym.service.runtime.create_and_run_compiler_service>` and pass in
    your class type:

    .. code-block:: python

        from compiler_gym.service import CompilationSession
        from compiler_gym.service import runtime

        class MyCompilationSession(CompilationSession):
            ...

        if __name__ == "__main__":
            runtime.create_and_run_compiler_service(MyCompilationSession)
    """

    compiler_version: str = ""
    """The compiler version."""

    action_spaces: List[ActionSpace] = []
    """A list of action spaces describing the capabilities of the compiler."""

    observation_spaces: List[ObservationSpace] = []
    """A list of feature vectors that this compiler provides."""

    def __init__(
        self, working_dir: Path, action_space: ActionSpace, benchmark: Benchmark
    ):
        """Start a CompilationSession.

        Subclasses should initialize the parent class first.

        :param working_dir: A directory on the local filesystem that can be used
            to store temporary files such as build artifacts.

        :param action_space: The action space to use.

        :param benchmark: The benchmark to use.
        """
        del action_space  # Subclasses must use this.
        del benchmark  # Subclasses must use this.
        self.working_dir = working_dir

    def apply_action(self, action: Action) -> Tuple[bool, Optional[ActionSpace], bool]:
        """Apply an action.

        :param action: The action to apply.

        :return: A tuple: :code:`(end_of_session, new_action_space,
        action_had_no_effect)`.
        """
        raise NotImplementedError

    def get_observation(self, observation_space: ObservationSpace) -> Observation:
        """Compute an observation.

        :param observation_space: The observation space.

        :return: An observation.
        """
        raise NotImplementedError

    def fork(self) -> "CompilationSession":
        """Create a copy of current session state.

        Implementing this method is optional.

        :return: A new CompilationSession with the same state.
        """
        # No need to override this if you are not adding support to fork().
        raise NotImplementedError("CompilationSession.fork() not supported")

    def handle_session_parameter(self, key: str, value: str) -> Optional[str]:
        """Handle a session parameter send by the frontend.

        Session parameters provide a method to send ad-hoc key-value messages to
        a compilation session through the :meth:`env.send_session_parameter()
        <compiler_gym.envs.CompilerEnv.send_session_parameter>` method. It us up
        to the client/service to agree on a common schema for encoding and
        decoding these parameters.

        Implementing this method is optional.

        :param key: The parameter key.

        :param value: The parameter value.

        :return: A string response message if the parameter was understood. Else
            :code:`None` to indicate that the message could not be interpretted.
        """
        pass

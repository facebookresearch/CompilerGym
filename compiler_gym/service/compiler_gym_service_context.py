# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CompilerGymServiceContext:
    """
    Execution context of a compiler gym service.

    This class encapsulates mutable state that is shared between all compilation
    sessions. An instance of this class is passed to every new
    CompilationSession.

    You may subclass CompilerGymServiceContext to add additional mutable state,
    or startup and shutdown routines. When overriding methods, subclasses should
    call the parent class implementation first.

    .. code-block:: python

        from compiler_gym.service import CompilationSession from
        compiler_gym.service import CompilerGymServiceContext from
        compiler_gym.service import runtime

        class MyServiceContext(CompilerGymServiceContext):
            ...

        class MyCompilationSession(CompilationSession):
            ...

        if __name__ == "__main__":
            runtime.create_and_run_compiler_service(
                MyCompilationSession, MyServiceContext,
            )
    """

    def __init__(self, working_directory: Path) -> None:
        """
        Initialize context.

        Called before any compilation sessions are created. Use this method to
        initialize any mutable state. If this routine returns an error, the
        service will terminate.
        """
        logger.debug("Initializing compiler service context")
        self.working_directory: Path = working_directory

    def shutdown(self) -> None:
        """
        Uninitialize context.

        Called after all compilation sessions have ended, before a service
        terminates. Use this method to perform tidying up. This method is always
        called, even if init() fails. If this routine returns an error, the
        service will terminate with a nonzero error code.
        """
        logger.debug("Closing compiler service context")

    def __enter__(self) -> "CompilerGymServiceContext":
        """Support 'with' syntax."""
        return self

    def __exit__(self, *args):
        """Support 'with' syntax."""
        self.shutdown()

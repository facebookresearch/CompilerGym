#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A CompilerGym service for GCC."""
import codecs
import hashlib
import json
import logging
import os
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.request import urlopen

from compiler_gym.envs.gcc import Gcc, Option
from compiler_gym.service import CompilationSession
from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    ByteSequenceSpace,
    ByteTensor,
    Event,
    Int64Range,
    Int64Tensor,
    ListSpace,
    NamedDiscreteSpace,
    ObservationSpace,
    Space,
    StringSpace,
)

logger = logging.getLogger(__name__)


def make_gcc_compilation_session(gcc_bin: str):
    """Create a class to represent a GCC compilation service.

    :param gcc_bin: Path to the gcc executable. This can a command name, like
        "gcc", or it can be path to the executable. Finally, if prefixed with
        "docker:" it can be the name of a docker image, e.g. "docker:gcc:11.2.0"
    """
    gcc = Gcc(gcc_bin)

    # The available actions
    actions = []

    # Actions that are small will have all their various choices made as
    # explicit actions.
    # Actions that are not small will have the abbility to increment the choice
    # by different amounts.
    for i, option in enumerate(gcc.spec.options):
        if len(option) < 10:
            for j in range(len(option)):
                actions.append(SimpleAction(option, i, j))
        if len(option) >= 10:
            actions.append(IncrAction(option, i, 1))
            actions.append(IncrAction(option, i, -1))
        if len(option) >= 50:
            actions.append(IncrAction(option, i, 10))
            actions.append(IncrAction(option, i, -10))
        if len(option) >= 500:
            actions.append(IncrAction(option, i, 100))
            actions.append(IncrAction(option, i, -100))
        if len(option) >= 5000:
            actions.append(IncrAction(option, i, 1000))
            actions.append(IncrAction(option, i, -1000))

    action_spaces_ = [
        ActionSpace(
            name="default",
            space=Space(
                named_discrete=NamedDiscreteSpace(name=[str(a) for a in actions]),
            ),
        ),
    ]

    observation_spaces_ = [
        # A string of the source code
        ObservationSpace(
            name="source",
            space=Space(string_value=StringSpace(length_range=Int64Range(min=0))),
            deterministic=True,
            platform_dependent=False,
            default_observation=Event(string_value=""),
        ),
        # A string of the rtl code
        ObservationSpace(
            name="rtl",
            space=Space(string_value=StringSpace(length_range=Int64Range(min=0))),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(string_value=""),
        ),
        # A string of the assembled code
        ObservationSpace(
            name="asm",
            space=Space(string_value=StringSpace(length_range=Int64Range(min=0))),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(string_value=""),
        ),
        # The size of the assembled code
        ObservationSpace(
            name="asm_size",
            space=Space(int64_value=Int64Range(min=-1)),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                int64_value=-1,
            ),
        ),
        # The hash of the assembled code
        ObservationSpace(
            name="asm_hash",
            space=Space(
                string_value=StringSpace(length_range=Int64Range(min=0, max=200)),
            ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(string_value=""),
        ),
        # Asm instruction counts - Counter as a JSON string
        ObservationSpace(
            name="instruction_counts",
            space=Space(
                string_value=StringSpace(length_range=Int64Range(min=0)),
            ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(string_value=""),
        ),
        # A bytes of the object code
        ObservationSpace(
            name="obj",
            space=Space(
                byte_sequence=ByteSequenceSpace(length_range=Int64Range(min=0)),
            ),
            deterministic=True,
            platform_dependent=False,
            default_observation=Event(byte_tensor=ByteTensor(shape=[0], value=b"")),
        ),
        # The size of the object code
        ObservationSpace(
            name="obj_size",
            space=Space(int64_value=Int64Range(min=-1)),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(
                int64_value=-1,
            ),
        ),
        # The hash of the object code
        ObservationSpace(
            name="obj_hash",
            space=Space(
                string_value=StringSpace(length_range=Int64Range(min=0, max=200)),
            ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(string_value=""),
        ),
        # A list of the choices. Each element corresponds to an option in the spec.
        # '-1' indicates that this is empty on the command line (e.g. if the choice
        # corresponding to the '-O' option is -1, then no -O flag will be emitted.)
        # If a nonnegative number if given then that particular choice is used
        # (e.g. for the -O flag, 5 means use '-Ofast' on the command line.)
        ObservationSpace(
            name="choices",
            space=Space(
                space_list=ListSpace(
                    space=[
                        Space(int64_value=Int64Range(min=0, max=len(option) - 1))
                        for option in gcc.spec.options
                    ]
                ),
            ),
        ),
        # The command line for compiling the object file as a string
        ObservationSpace(
            name="command_line",
            space=Space(
                string_value=StringSpace(length_range=Int64Range(min=0, max=200)),
            ),
            deterministic=True,
            platform_dependent=True,
            default_observation=Event(string_value=""),
        ),
    ]

    class GccCompilationSession(CompilationSession):
        """A GCC interactive compilation session."""

        compiler_version: str = gcc.spec.version
        action_spaces = action_spaces_
        observation_spaces = observation_spaces_

        def __init__(
            self,
            working_directory: Path,
            action_space: ActionSpace,
            benchmark: Benchmark,
        ):
            super().__init__(working_directory, action_space, benchmark)
            # The benchmark being used
            self.benchmark = benchmark
            # Timeout value for compilation (in seconds)
            self._timeout = None
            # The source code
            self._source = None
            # The rtl code
            self._rtl = None
            # The assembled code
            self._asm = None
            # Size of the assembled code
            self._asm_size = None
            # Hash of the assembled code
            self._asm_hash = None
            # The object binary
            self._obj = None
            # size of the object binary
            self._obj_size = None
            # Hash of the object binary
            self._obj_hash = None
            # Set the path to the GCC executable
            self._gcc_bin = "gcc"
            # Initially the choices and the spec, etc are empty. They will be
            # initialised lazily
            self._choices = None

        @property
        def num_actions(self) -> int:
            return len(self.action_spaces[0].space.named_discrete.name)

        @property
        def choices(self) -> List[int]:
            if self._choices is None:
                self._choices = [-1] * len(gcc.spec.options)
            return self._choices

        @choices.setter
        def choices(self, value: List[int]):
            self._choices = value

        @property
        def source(self) -> str:
            """Get the benchmark source"""
            self.prepare_files()
            return self._source

        @property
        def rtl(self) -> bytes:
            """Get the RTL code"""
            self.dump_rtl()
            return self._rtl

        @property
        def asm(self) -> bytes:
            """Get the assembled code"""
            self.assemble()
            return self._asm

        @property
        def asm_size(self) -> int:
            """Get the assembled code size"""
            self.assemble()
            return self._asm_size

        @property
        def asm_hash(self) -> str:
            """Get the assembled code hash"""
            self.assemble()
            return self._asm_hash

        @property
        def instruction_counts(self) -> str:
            """Get the instuction counts as a JSON string"""
            self.assemble()
            insn_pat = re.compile("\t([a-zA-Z-0-9.-]+).*")
            insn_cnts = Counter()
            lines = self._asm.split("\n")
            for line in lines:
                m = insn_pat.fullmatch(line)
                if m:
                    insn_cnts[m.group(1)] += 1

            return json.dumps(insn_cnts)

        @property
        def obj(self) -> bytes:
            """Get the compiled code"""
            self.compile()
            return self._obj

        @property
        def obj_size(self) -> int:
            """Get the compiled code size"""
            self.compile()
            return self._obj_size

        @property
        def obj_hash(self) -> str:
            """Get the compiled code hash"""
            self.compile()
            return self._obj_hash

        @property
        def src_path(self) -> Path:
            """Get the path to the source file"""
            return self.working_dir / "src.c"

        @property
        def obj_path(self) -> Path:
            """Get the path to object file"""
            return self.working_dir / "obj.o"

        @property
        def asm_path(self) -> Path:
            """Get the path to the assembly"""
            return self.working_dir / "asm.s"

        @property
        def rtl_path(self) -> Path:
            """Get the path to the rtl"""
            return self.working_dir / "rtl.lsp"

        def obj_command_line(
            self, src_path: Path = None, obj_path: Path = None
        ) -> List[str]:
            """Get the command line to create the object file.
            The 'src_path' and 'obj_path' give the input and output paths. If not
            set, then they are taken from 'self.src_path' and 'self.obj_path'. This
            is useful for printing where the actual paths are not important."""
            src_path = src_path or self.src_path
            obj_path = obj_path or self.obj_path
            # Gather the choices as strings
            opts = [
                option[choice]
                for option, choice in zip(gcc.spec.options, self.choices)
                if choice >= 0
            ]
            cmd_line = opts + ["-w", "-c", src_path, "-o", obj_path]
            return cmd_line

        def asm_command_line(
            self, src_path: Path = None, asm_path: Path = None
        ) -> List[str]:
            """Get the command line to create the assembly file.
            The 'src_path' and 'asm_path' give the input and output paths. If not
            set, then they are taken from 'self.src_path' and 'self.obj_path'. This
            is useful for printing where the actual paths are not important."""
            src_path = src_path or self.src_path
            asm_path = asm_path or self.asm_path
            opts = [
                option[choice]
                for option, choice in zip(gcc.spec.options, self.choices)
                if choice >= 0
            ]
            cmd_line = opts + ["-w", "-S", src_path, "-o", asm_path]
            return cmd_line

        def rtl_command_line(
            self, src_path: Path = None, rtl_path: Path = None, asm_path: Path = None
        ) -> List[str]:
            """Get the command line to create the rtl file - might as well do the
            asm at the same time.
            The 'src_path', 'rtl_path', 'asm_path' give the input and output paths. If not
            set, then they are taken from 'self.src_path' and 'self.obj_path'. This
            is useful for printing where the actual paths are not important."""
            src_path = src_path or self.src_path
            rtl_path = rtl_path or self.rtl_path
            asm_path = asm_path or self.asm_path
            opts = [
                option[choice]
                for option, choice in zip(gcc.spec.options, self.choices)
                if choice >= 0
            ]
            cmd_line = opts + [
                "-w",
                "-S",
                src_path,
                f"-fdump-rtl-dfinish={rtl_path}",
                "-o",
                asm_path,
            ]
            return cmd_line

        def prepare_files(self):
            """Copy the source to the working directory."""
            if not self._source:
                if self.benchmark.program.contents:
                    self._source = self.benchmark.program.contents.decode()
                else:
                    with urlopen(self.benchmark.program.uri) as r:
                        self._source = r.read().decode()

                with open(self.src_path, "w") as f:
                    print(self._source, file=f)

        def compile(self) -> Optional[str]:
            """Compile the benchmark"""
            if not self._obj:
                self.prepare_files()
                logger.debug(
                    "Compiling: %s", " ".join(map(str, self.obj_command_line()))
                )
                gcc(
                    *self.obj_command_line(),
                    cwd=self.working_dir,
                    timeout=self._timeout,
                )
                with open(self.obj_path, "rb") as f:
                    # Set the internal variables
                    self._obj = f.read()
                    self._obj_size = os.path.getsize(self.obj_path)
                    self._obj_hash = hashlib.md5(self._obj).hexdigest()

        def assemble(self) -> Optional[str]:
            """Assemble the benchmark"""
            if not self._asm:
                self.prepare_files()
                logger.debug(
                    "Assembling: %s", " ".join(map(str, self.asm_command_line()))
                )
                gcc(
                    *self.asm_command_line(),
                    cwd=self.working_dir,
                    timeout=self._timeout,
                )
                with open(self.asm_path, "rb") as f:
                    # Set the internal variables
                    asm_bytes = f.read()
                    self._asm = asm_bytes.decode()
                    self._asm_size = os.path.getsize(self.asm_path)
                    self._asm_hash = hashlib.md5(asm_bytes).hexdigest()

        def dump_rtl(self) -> Optional[str]:
            """Dump the RTL (and assemble the benchmark)"""
            if not self._rtl:
                self.prepare_files()
                logger.debug(
                    "Dumping RTL: %s", " ".join(map(str, self.rtl_command_line()))
                )
                gcc(
                    *self.rtl_command_line(),
                    cwd=self.working_dir,
                    timeout=self._timeout,
                )
                with open(self.asm_path, "rb") as f:
                    # Set the internal variables
                    asm_bytes = f.read()
                    self._asm = asm_bytes.decode()
                    self._asm_size = os.path.getsize(self.asm_path)
                    self._asm_hash = hashlib.md5(asm_bytes).hexdigest()
                with open(self.rtl_path, "rb") as f:
                    # Set the internal variables
                    rtl_bytes = f.read()
                    self._rtl = rtl_bytes.decode()

        def reset_cached(self):
            """Reset the cached values"""
            self._obj = None
            self._obj_size = None
            self._obj_hash = None
            self._rtl = None
            self._asm = None
            self._asm_size = None
            self._asm_hash = None

        def apply_action(
            self, action_proto: Event
        ) -> Tuple[bool, Optional[ActionSpace], bool]:
            """Apply an action."""
            if not action_proto.HasField("int64_value"):
                raise ValueError("Invalid action, int64_value expected.")

            choice_index = action_proto.int64_value
            if choice_index < 0 or choice_index >= self.num_actions:
                raise ValueError("Out-of-range")

            # Get the action
            action = actions[choice_index]
            # Apply the action to this session and check if we changed anything
            old_choices = self.choices.copy()
            action(self)
            logger.debug("Applied action %s", action)

            # Reset the internal variables if this action has caused a change in the
            # choices
            if old_choices != self.choices:
                self.reset_cached()

            # The action has not changed anything yet. That waits until an
            # observation is taken
            return False, None, False

        def get_observation(self, observation_space: ObservationSpace) -> Event:
            """Get one of the observations"""
            if observation_space.name == "source":
                return Event(string_value=self.source or "")
            elif observation_space.name == "rtl":
                return Event(string_value=self.rtl or "")
            elif observation_space.name == "asm":
                return Event(string_value=self.asm or "")
            elif observation_space.name == "asm_size":
                return Event(int64_value=self.asm_size or -1)
            elif observation_space.name == "asm_hash":
                return Event(string_value=self.asm_hash or "")
            elif observation_space.name == "instruction_counts":
                return Event(string_value=self.instruction_counts or "{}")
            elif observation_space.name == "obj":
                value = self.obj or b""
                return Event(byte_tensor=ByteTensor(shape=[len(value)], value=value))
            elif observation_space.name == "obj_size":
                return Event(int64_value=self.obj_size or -1)
            elif observation_space.name == "obj_hash":
                return Event(string_value=self.obj_hash or "")
            elif observation_space.name == "choices":
                observation = Event(
                    int64_tensor=Int64Tensor(
                        shape=[len(self.choices)], value=self.choices
                    )
                )
                return observation
            elif observation_space.name == "command_line":
                return Event(
                    string_value=gcc.bin
                    + " "
                    + " ".join(map(str, self.obj_command_line("src.c", "obj.o")))
                )
            else:
                raise KeyError(observation_space.name)

        def handle_session_parameter(self, key: str, value: str) -> Optional[str]:
            if key == "gcc_spec":
                return codecs.encode(pickle.dumps(gcc.spec), "base64").decode()
            elif key == "choices":
                choices = list(map(int, value.split(",")))
                assert len(choices) == len(gcc.spec.options)
                assert all(
                    -1 <= p <= len(gcc.spec.options[i]) for i, p in enumerate(choices)
                )
                if choices != self.choices:
                    self.choices = choices
                    self.reset_cached()
                return ""
            elif key == "timeout":
                self._timeout = None if value == "" else int(value)
                return ""
            return None

    return GccCompilationSession


class Action:
    """An action is applying a choice to an option"""

    def __init__(self, option: Option, option_index: int):
        """The option and its index in the option list.  We need the index to
        match it with the corresponding choice later during the application of
        the action."""
        self.option = option
        self.option_index = option_index

    def __call__(self, session: "GccCompilationSession"):  # noqa
        """Apply the action to the session."""
        raise NotImplementedError()

    def __str__(self) -> str:
        raise NotImplementedError()


class SimpleAction(Action):
    """A simple action just sets the choice directly.
    The choice_index describes which choice to apply."""

    def __init__(self, option: Option, option_index: int, choice_index: int):
        super().__init__(option, option_index)
        self.choice_index = choice_index

    def __call__(self, session: "GccCompilationSession"):  # noqa
        session.choices[self.option_index] = self.choice_index

    def __str__(self) -> str:
        return self.option[self.choice_index]


class IncrAction(Action):
    """An action that increments a choice by an amount."""

    def __init__(self, option: Option, option_index: int, choice_incr: int):
        super().__init__(option, option_index)
        self.choice_incr = choice_incr

    def __call__(self, session: "GccCompilationSession"):  # noqag
        choice = session.choices[self.option_index]
        choice += self.choice_incr
        if choice < -1:
            choice = -1
        if choice >= len(self.option):
            choice = len(self.option) - 1
        session.choices[self.option_index] = choice

    def __str__(self) -> str:
        return f"{self.option}[{self.choice_incr:+}]"

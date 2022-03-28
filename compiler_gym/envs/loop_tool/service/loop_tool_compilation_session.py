# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Define the loop_tool environment."""
import logging
import time
from functools import reduce
from pathlib import Path
from typing import Optional, Tuple

import loop_tool_py as lt
import numpy as np
import pkg_resources

from compiler_gym.service import CompilationSession, EnvironmentNotSupported
from compiler_gym.service.proto import (
    ActionSpace,
    Benchmark,
    DoubleRange,
    Event,
    Int64Box,
    Int64Range,
    Int64Tensor,
    NamedDiscreteSpace,
    ObservationSpace,
    Space,
    StringSpace,
)

logger = logging.getLogger(__name__)


class LoopToolCompilationSession(CompilationSession):
    """Represents an instance of an interactive loop_tool session."""

    compiler_version: str = pkg_resources.get_distribution("loop-tool-py").version

    # keep it simple for now: 1 variable, 1 nest
    action_spaces = [
        ActionSpace(
            name="simple",
            space=Space(
                # shift around a single pre-split order, changing the size of splits
                named_discrete=NamedDiscreteSpace(
                    name=["toggle_mode", "up", "down", "toggle_thread"],
                ),
            ),
        ),
        ActionSpace(
            name="split",
            space=Space(
                # potentially define new splits
                named_discrete=NamedDiscreteSpace(
                    name=["toggle_mode", "up", "down", "toggle_thread", "split"],
                ),
            ),
        ),
    ]

    observation_spaces = [
        ObservationSpace(
            name="flops",
            space=Space(double_value=DoubleRange()),
            deterministic=False,
            platform_dependent=True,
            default_observation=Event(
                double_value=0,
            ),
        ),
        ObservationSpace(
            name="loop_tree",
            space=Space(
                string_value=StringSpace(length_range=Int64Range(min=0)),
            ),
            deterministic=True,
            platform_dependent=False,
            default_observation=Event(
                string_value="",
            ),
        ),
        ObservationSpace(
            name="action_state",
            space=Space(
                int64_box=Int64Box(
                    low=Int64Tensor(shape=[1], value=[0]),
                    high=Int64Tensor(shape=[1], value=[2 ** 36]),
                ),
            ),
            deterministic=True,
            platform_dependent=False,
            default_observation=Event(
                int64_tensor=Int64Tensor(shape=[1], value=[0]),
            ),
        ),
    ]

    def __init__(
        self, working_directory: Path, action_space: ActionSpace, benchmark: Benchmark
    ):
        super().__init__(working_directory, action_space, benchmark)
        self.action_space = action_space
        if "cuda" in benchmark.uri:
            self.backend = "cuda"
            lt.set_default_hardware("cuda")
        else:
            self.backend = "cpu"
        if self.backend not in lt.backends():
            raise EnvironmentNotSupported(
                f"Failed to load {self.backend} dataset for loop_tool.  Have you installed all required dependecies?  See <https://facebookresearch.github.io/CompilerGym/envs/loop_tool.html#installation> for details. "
            )
        self.ir = lt.IR()
        self.var = self.ir.create_var("a")
        r0 = self.ir.create_node("read", [], [self.var])
        r1 = self.ir.create_node("read", [], [self.var])
        add = self.ir.create_node("add", [r0, r1], [self.var])
        w = self.ir.create_node("write", [add], [self.var])
        self.ir.set_inputs([r0, r1])
        self.ir.set_outputs([w])
        self.size = int(benchmark.uri.split("/")[-1])
        self.Ap = np.random.randn(self.size)
        self.Bp = np.random.randn(self.size)
        self.order = [(self.size, 0), (1, 0), (1, 0)]
        self.thread = [1, 0, 0]
        self.cursor = 0
        self.mode = "size"
        logger.info("Started a compilation session for %s", benchmark.uri)

    def resize(self, increment):
        """
        The idea is pull from or add to the parent loop.

        Three mutations possible to any size:
        A) x, y -> x + 1, 0
          remove the tail, increase loop size, shrink parent
        B) x, y -> x, 0
          only remove the tail, add to parent
        C) x, 0 -> x - 1, 0
          if no tail, shrink the loop size, increase parent

        note: this means tails can never exist on innermost loops. this makes good sense :)

        A)

        [(a, b), (x, y), ...k] -> [(a', b'), (x + 1, 0), ...k]
        a * (x * k + y) + b = a' * (x + 1) * k + b'
        a' = (a * (x * k + y) + b) // ((x + 1) * k)
        b' = "                   " %  "           "

        B)

        [(a, b), (x, y), ...k] -> [(a', b'), (x, 0), ...k]
        a * (x * k + y) + b = a' * (x) * k + b'
        a' = (a * (x * k + y) + b) // ((x) * k)
        b' = "                   " %  "           "

        C)

        [(a, b), (x, y), ...k] -> [(a', b'), (x - 1, 0), ...k]
        a * (x * k + y) + b = a' * (x - 1) * k + b'
        a' = (a * (x * k + y) + b) // ((x - 1) * k)
        b' = "                   " %  "           "

        example interaction model:
        1. cursor = 1        [1024, 1, 1]
        2. up                [512, 2, 1]
        3. up                [(341,1), 3, 1]
        4. up                [256, 4, 1]
        5. cursor = 2, up    [256, 2, 2]
        6. up                [256, (1, 1), 3]
        7. cursor = 1, down  [(341, 1), 1, 3]
        8. cursor = 2, down  [(341, 1), (1, 1), 2]
        9. cursor = 1, down  [512, 1, 2]"""
        if self.cursor == 0:
            return
        parent_size = self.order[self.cursor - 1]
        a = parent_size[0]
        b = parent_size[1]
        size = self.order[self.cursor]
        x = size[0]
        y = size[1]

        def lam(v, x):
            return v * x[0] + x[1]

        k = reduce(lam, self.order[self.cursor + 1 :][::-1], 1)
        if increment == -1 and y:
            increment = 0
        if (x + increment) < 1:
            return
        if (x + increment) > self.size:
            return
        n = a * x * k + b
        d = (x + increment) * k
        a_ = n // d
        b_ = n % d
        if a_ < 1:
            return
        if a_ > self.size:
            return
        self.order[self.cursor - 1] = (a_, b_)
        self.order[self.cursor] = (x + increment, 0)
        end_size = reduce(lam, self.order[::-1], 1)
        assert (
            end_size == self.size
        ), f"{end_size} != {self.size} ({a}, {b}), ({x}, {y}) -> ({a_}, {b_}), ({x + increment}, 0)"

    def apply_action(self, action: Event) -> Tuple[bool, Optional[ActionSpace], bool]:
        if not action.HasField("int64_value"):
            raise ValueError("Invalid action. int64_value expected.")

        choice_index = action.int64_value
        if choice_index < 0 or choice_index >= len(
            self.action_space.space.named_discrete.name
        ):
            raise ValueError("Out-of-range")

        logger.info("Applied action %d", choice_index)

        act = self.action_space.space.named_discrete.name[choice_index]
        if self.mode not in ["size", "select"]:
            raise RuntimeError("Invalid mode set: {}".format(self.mode))
        if act == "toggle_mode":
            if self.mode == "size":
                self.mode = "select"
            elif self.mode == "select":
                self.mode = "size"
        if act == "toggle_thread":
            self.thread[self.cursor] = not self.thread[self.cursor]
        if act == "down":
            # always loop around
            if self.mode == "size":
                self.resize(-1)
            elif self.mode == "select":
                next_cursor = (self.cursor - 1) % len(self.order)
                self.cursor = next_cursor
        if act == "up":
            # always loop around
            if self.mode == "size":
                self.resize(1)
            elif self.mode == "select":
                next_cursor = (self.cursor + 1) % len(self.order)
                self.cursor = next_cursor

        return False, None, False

    def lower(self):
        for n in self.ir.nodes:
            o = [(self.var, k) for k in self.order]
            self.ir.set_order(n, o)
            # always disable innermost
            self.ir.disable_reuse(n, len(o) - 1)
        loop_tree = lt.LoopTree(self.ir)
        parallel = set()
        t = loop_tree.roots[0]
        for b in self.thread:
            if b:
                parallel.add(t)
                if self.backend == "cpu":
                    loop_tree.annotate(t, "cpu_parallel")
            t = loop_tree.children(t)[0]
        return loop_tree, parallel

    def flops(self):
        loop_tree, parallel = self.lower()
        if self.backend == "cuda":
            c = lt.cuda(loop_tree, parallel)
        else:
            c = lt.cpu(loop_tree)
        A = lt.Tensor(self.size)
        B = lt.Tensor(self.size)
        C = lt.Tensor(self.size)
        A.set(self.Ap)
        B.set(self.Bp)
        iters = 1000
        warmup = 50
        for i in range(warmup):
            c([A, B, C])
        t = time.time()
        for i in range(iters - 1):
            c([A, B, C], False)
        c([A, B, C])
        t_ = time.time()
        flops = self.size * iters / (t_ - t) / 1e9
        return flops

    def get_observation(self, observation_space: ObservationSpace) -> Event:
        if observation_space.name == "action_state":
            # cursor, (size, tail)
            o = self.order[self.cursor]
            return Event(
                int64_tensor=Int64Tensor(shape=[3], value=[self.cursor, o[0], o[1]])
            )
        elif observation_space.name == "flops":
            return Event(double_value=self.flops())
        elif observation_space.name == "loop_tree":
            loop_tree, parallel = self.lower()
            return Event(
                string_value=loop_tree.dump(
                    lambda x: "[thread]" if x in parallel else ""
                )
            )
        else:
            raise KeyError(observation_space.name)

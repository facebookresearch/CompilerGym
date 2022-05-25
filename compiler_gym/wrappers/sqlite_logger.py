# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""This module implements a wrapper that logs state transitions to an sqlite
database.
"""
import logging
import pickle
import sqlite3
import zlib
from pathlib import Path
from time import time
from typing import Iterable, Optional, Union

import numpy as np

from compiler_gym.envs import LlvmEnv
from compiler_gym.spaces import Reward
from compiler_gym.util.gym_type_hints import ActionType
from compiler_gym.util.timer import Timer, humanize_duration
from compiler_gym.views import ObservationSpaceSpec
from compiler_gym.wrappers import CompilerEnvWrapper

DB_CREATION_SCRIPT = """
CREATE TABLE IF NOT EXISTS States (
  benchmark_uri TEXT NOT NULL,         -- The URI of the benchmark.
  done INTEGER NOT NULL,               -- 0 = False, 1 = True.
  ir_instruction_count_oz_reward REAL NULLABLE,
  state_id TEXT NOT NULL,              -- 40-char sha1.
  actions TEXT NOT NULL,               -- Decode: [int(x) for x in field.split()]
  PRIMARY KEY (benchmark_uri, actions),
  FOREIGN KEY (state_id) REFERENCES Observations(state_id) ON UPDATE CASCADE
);

CREATE TABLE IF NOT EXISTS Observations (
    state_id TEXT NOT NULL,            -- 40-char sha1.
    ir_instruction_count INTEGER NOT NULL,
    compressed_llvm_ir BLOB NOT NULL,           -- Decode: zlib.decompress(...)
    pickled_compressed_programl BLOB NOT NULL,  -- Decode: pickle.loads(zlib.decompress(...))
    autophase TEXT NOT NULL,                    -- Decode: np.array([int(x) for x in field.split()], dtype=np.int64)
    instcount TEXT NOT NULL,                    -- Decode: np.array([int(x) for x in field.split()], dtype=np.int64)
    PRIMARY KEY (state_id)
);
"""


class SynchronousSqliteLogger(CompilerEnvWrapper):
    """A wrapper for an LLVM environment that logs all transitions to an sqlite
    database.

    Wrap an existing LLVM environment and then use it as per normal:

        >>> env = SynchronousSqliteLogger(
        ...     env=gym.make("llvm-autophase-ic-v0"),
        ...     db_path="example.db",
        ... )

    Connect to the database file you specified:

    .. code-block::
        $ sqlite3 example.db

    There are two tables:

    1. States: records every unique combination of benchmark + actions. For each
       entry, records an identifying state ID, the episode reward, and whether
       the episode is terminated:

    .. code-block::

        sqlite> .mode markdown
        sqlite> .headers on
        sqlite> select * from States limit 5;
        |      benchmark_uri       | done | ir_instruction_count_oz_reward |                                 state_id | actions |
        |--------------------------|------|--------------------------------|------------------------------------------|----------------|
        | generator://csmith-v0/99 | 0    | 0.0                            | d625b874e58f6d357b816e21871297ac5c001cf0 |                |
        | generator://csmith-v0/99 | 0    | 0.0                            | d625b874e58f6d357b816e21871297ac5c001cf0 | 31             |
        | generator://csmith-v0/99 | 0    | 0.0                            | 52f7142ef606d8b1dec2ff3371c7452c8d7b81ea | 31 116         |
        | generator://csmith-v0/99 | 0    | 0.268005818128586              | d8c05bd41b7a6c6157b6a8f0f5093907c7cc7ecf | 31 116 103     |
        | generator://csmith-v0/99 | 0    | 0.288621664047241              | c4d7ecd3807793a0d8bc281104c7f5a8aa4670f9 | 31 116 103 109 |

    2. Observations: records pickled, compressed, and text observation values
       for each unique state.

    Caveats of this implementation:

    1. Only :class:`LlvmEnv <compiler_gym.envs.LlvmEnv>` environments may be
       wrapped.

    2. The wrapped environment must have an observation space and reward space
       set.

    3. The observation spaces and reward spaces that are logged to database
       are hardcoded. To change what is recorded, you must copy and modify this
       implementation.

    4. Writing to the database is synchronous and adds significant overhead to
       the compute cost of the environment.
    """

    def __init__(
        self,
        env: LlvmEnv,
        db_path: Path,
        commit_frequency_in_seconds: int = 300,
        max_step_buffer_length: int = 5000,
    ):
        """Constructor.

        :param env: The environment to wrap.

        :param db_path: The path of the database to log to. This file may
            already exist. If it does, new entries are appended. If the files
            does not exist, it is created.

        :param commit_frequency_in_seconds: The maximum amount of time to elapse
            before writing pending logs to the database.

        :param max_step_buffer_length: The maximum number of calls to
            :code:`step()` before writing pending logs to the database.
        """
        super().__init__(env)
        if not hasattr(env, "unwrapped"):
            raise TypeError("Requires LlvmEnv base environment")
        if not isinstance(self.unwrapped, LlvmEnv):
            raise TypeError("Requires LlvmEnv base environment")
        db_path.parent.mkdir(exist_ok=True, parents=True)
        self.connection = sqlite3.connect(str(db_path))
        self.cursor = self.connection.cursor()
        self.commit_frequency = commit_frequency_in_seconds
        self.max_step_buffer_length = max_step_buffer_length

        self.cursor.executescript(DB_CREATION_SCRIPT)
        self.connection.commit()
        self.last_commit = time()

        self.observations_buffer = {}
        self.step_buffer = []

        # House keeping notice: Keep these lists in sync with record().
        self._observations = [
            self.env.observation.spaces["IrSha1"],
            self.env.observation.spaces["Ir"],
            self.env.observation.spaces["Programl"],
            self.env.observation.spaces["Autophase"],
            self.env.observation.spaces["InstCount"],
            self.env.observation.spaces["IrInstructionCount"],
        ]
        self._rewards = [
            self.env.reward.spaces["IrInstructionCountOz"],
            self.env.reward.spaces["IrInstructionCount"],
        ]
        self._reward_totals = np.zeros(len(self._rewards))

    def flush(self) -> None:
        """Flush the buffered steps and observations to database."""
        n_steps, n_observations = len(self.step_buffer), len(self.observations_buffer)

        # Nothing to flush.
        if not n_steps:
            return

        with Timer() as flush_time:
            # House keeping notice: Keep these statements in sync with record().
            self.cursor.executemany(
                "INSERT OR IGNORE INTO States VALUES (?, ?, ?, ?, ?)",
                self.step_buffer,
            )
            self.cursor.executemany(
                "INSERT OR IGNORE INTO Observations VALUES (?, ?, ?, ?, ?, ?)",
                ((k, *v) for k, v in self.observations_buffer.items()),
            )
            self.step_buffer = []
            self.observations_buffer = {}

            self.connection.commit()

        logging.info(
            "Wrote %d state records and %d observations in %s. Last flush %s ago",
            n_steps,
            n_observations,
            flush_time,
            humanize_duration(time() - self.last_commit),
        )
        self.last_commit = time()

    def reset(self, *args, **kwargs):
        observation = self.env.reset(*args, **kwargs)
        observations, rewards, done, info = self.env.multistep(
            actions=[],
            observation_spaces=self._observations,
            reward_spaces=self._rewards,
        )
        assert not done, f"reset() failed! {info}"
        self._reward_totals = np.array(rewards, dtype=np.float32)
        rewards = self._reward_totals
        self._record(
            actions=self.actions,
            observations=observations,
            rewards=self._reward_totals,
            done=False,
        )
        return observation

    def step(
        self,
        action: ActionType,
        observation_spaces: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        reward_spaces: Optional[Iterable[Union[str, Reward]]] = None,
        observations: Optional[Iterable[Union[str, ObservationSpaceSpec]]] = None,
        rewards: Optional[Iterable[Union[str, Reward]]] = None,
    ):
        assert self.observation_space, "No observation space set"
        assert self.reward_space, "No reward space set"
        assert (
            observation_spaces is None
        ), "SynchronousSqliteLogger does not support observation_spaces"
        assert (
            reward_spaces is None
        ), "SynchronousSqliteLogger does not support reward_spaces"
        assert (
            observations is None
        ), "SynchronousSqliteLogger does not support observations"
        assert rewards is None, "SynchronousSqliteLogger does not support rewards"

        observations, rewards, done, info = self.env.step(
            action=action,
            observation_spaces=self._observations + [self.observation_space_spec],
            reward_spaces=self._rewards + [self.reward_space],
        )
        self._reward_totals += rewards[:-1]
        self._record(
            actions=self.actions,
            observations=observations[:-1],
            rewards=self._reward_totals,
            done=done,
        )
        return observations[-1], rewards[-1], done, info

    def _record(self, actions, observations, rewards, done) -> None:
        state_id, ir, programl, autophase, instcount, instruction_count = observations
        instruction_count_reward = float(rewards[0])

        self.step_buffer.append(
            (
                str(self.benchmark.uri),
                1 if done else 0,
                instruction_count_reward,
                state_id,
                " ".join(str(x) for x in actions),
            )
        )

        self.observations_buffer[state_id] = (
            instruction_count,
            zlib.compress(ir.encode("utf-8")),
            zlib.compress(pickle.dumps(programl)),
            " ".join(str(x) for x in autophase),
            " ".join(str(x) for x in instcount),
        )

        if (
            len(self.step_buffer) >= self.max_step_buffer_length
            or time() - self.last_commit >= self.commit_frequency
        ):
            self.flush()

    def close(self):
        self.flush()
        self.env.close()

    def fork(self):
        raise NotImplementedError

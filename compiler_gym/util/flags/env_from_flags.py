# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A consistent way to interpret a user-specified environment from commandline flags."""
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union

import gym
from absl import app, flags

from compiler_gym.envs import CompilerEnv
from compiler_gym.service import ConnectionOpts
from compiler_gym.service.proto import Benchmark
from compiler_gym.util.registration import COMPILER_GYM_ENVS

flags.DEFINE_string(
    "env",
    None,
    "The name of an environment to use. The environment must be registered with "
    "gym.",
)
flags.DEFINE_string(
    "service",
    None,
    "If set, this specifies the hostname and port of a service to connect to, "
    "rather than creating a new environment. Use the format <hostname>:<port>. "
    "Supersedes --local_service_binary.",
)
flags.DEFINE_string(
    "local_service_binary",
    None,
    "If set, this specifies the path of a local service binary to run to "
    "provide the environment service, rather than the default service binary.",
)
flags.DEFINE_string(
    "observation",
    None,
    "The name of a observation space to use. If set, this overrides any "
    "default set by the environment.",
)
flags.DEFINE_string(
    "reward",
    None,
    "The name of a reward space to use. If set, this overrides any default "
    "set by the environment.",
)
flags.DEFINE_boolean(
    "ls_env",
    False,
    "Print the list of available environments that can be passed to --env and exit.",
)

# Connection settings.

flags.DEFINE_float(
    "service_rpc_call_max_seconds",
    300,
    "Service configuration option. Limits the maximum number of seconds to wait "
    "for a service RPC to return a response.",
)
flags.DEFINE_float(
    "service_init_max_seconds",
    10,
    "Service configuration option. Limits the maximum number of seconds to wait "
    "to establish a connection to a service.",
)
flags.DEFINE_integer(
    "service_init_max_attempts",
    5,
    "Service configuration option. Limits the maximum number of attempts to "
    "initialize a service.",
)
flags.DEFINE_float(
    "local_service_port_init_max_seconds",
    10,
    "Service configuration option. Limits the maximum number of seconds to wait "
    "for a local service to write a port.txt file on initialization.",
)
flags.DEFINE_float(
    "local_service_exit_max_seconds",
    10,
    "Service configuration option. Limits the maximum number of seconds to wait "
    "for a local service to terminate on close.",
)
flags.DEFINE_float(
    "service_rpc_init_max_seconds",
    3,
    "Service configuration option. Limits the number of seconds to wait for an "
    "RPC connection to establish on initialization.",
)

FLAGS = flags.FLAGS


def connection_settings_from_flags(
    service_url: str = None, local_service_binary: Path = None
) -> ConnectionOpts:
    """Returns either the name of the benchmark, or a Benchmark message."""
    return ConnectionOpts(
        rpc_call_max_seconds=FLAGS.service_rpc_call_max_seconds,
        init_max_seconds=FLAGS.service_init_max_seconds,
        init_max_attempts=FLAGS.service_init_max_attempts,
        local_service_port_init_max_seconds=FLAGS.local_service_port_init_max_seconds,
        local_service_exit_max_seconds=FLAGS.local_service_exit_max_seconds,
        rpc_init_max_seconds=FLAGS.service_rpc_init_max_seconds,
    )


def env_from_flags(benchmark: Optional[Union[str, Benchmark]] = None) -> CompilerEnv:
    if FLAGS.ls_env:
        print("\n".join(sorted(COMPILER_GYM_ENVS)))
        sys.exit(0)

    connection_settings = connection_settings_from_flags()

    if not FLAGS.env:
        raise app.UsageError("--env must be set")

    init_opts = {
        "benchmark": benchmark,
        "connection_settings": connection_settings,
    }

    if FLAGS.local_service_binary:
        init_opts["service"] = Path(FLAGS.service)

    if FLAGS.service:
        init_opts["service"] = FLAGS.service

    env = gym.make(FLAGS.env, **init_opts)

    if FLAGS.observation:
        env.observation_space = FLAGS.observation
    if FLAGS.reward:
        env.reward_space = FLAGS.reward

    return env


@contextmanager
def env_session_from_flags(
    benchmark: Optional[Union[str, Benchmark]] = None
) -> CompilerEnv:
    with env_from_flags(benchmark=benchmark) as env:
        yield env

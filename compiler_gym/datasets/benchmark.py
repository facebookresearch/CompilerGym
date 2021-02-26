# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import re
from concurrent.futures import as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from compiler_gym.service.proto import Benchmark as BenchmarkProto
from compiler_gym.service.proto import File
from compiler_gym.util import thread_pool
from compiler_gym.validation_result import ValidationError

# A validation callback is a function that takes a single CompilerEnv instance
# as its argument and returns an iterable sequence of zero or more
# ValidationError tuples.
ValidationCallback = Callable[["CompilerEnv"], Iterable[ValidationError]]  # noqa: F821


# Regular expression that matches the full two-part URI prefix of a dataset:
#     <protocol>://<dataset>
#
# E.g. "benchmark://foo-v0".
DATASET_NAME_RE = re.compile(
    r"(?P<dataset>(?P<dataset_protocol>[a-zA-z0-9-_]+)://(?P<dataset_name>[a-zA-z0-9-_]+-v(?P<dataset_version>[0-9]+)))"
)

# Regular expression that matches the full three-part format of a benchmark URI:
#     <protocol>://<dataset>/<id>
#
# E.g. "benchmark://foo-v0/" or "benchmark://foo-v0/program".
BENCHMARK_URI_RE = re.compile(
    r"(?P<dataset>(?P<dataset_protocol>[a-zA-z0-9-_]+)://(?P<dataset_name>[a-zA-z0-9-_]+-v(?P<dataset_version>[0-9]+)))/(?P<benchmark_name>[^\s]*)$"
)


def resolve_uri_protocol(uri: str):
    if "://" not in uri:
        return f"benchmark://{uri}"
    return uri


class Benchmark(object):
    """A program that is used as input to a compiler environment.

    A benchmark is a program that can be used by a :class:`CompilerEnv
    <compiler_gym.envs.CompilerEnv>` as a program to optimize. A benchmark
    comprises the data that is fed into the compiler, and a URI that is used to
    identify instances of that benchmark.

    Benchmarks are not normally instantiated directly. Instead, a benchmark is
    generated using :meth:`env.datasets.benchmark()
    <compiler_gym.datasets.Datasets.benchmark>`. Available benchmarks can be
    queried using :meth:`env.datasets.benchmark_uris()
    <compiler_gym.datasets.Datasets.benchmark_uris>`. Compiler environments may
    provide helper functions for generating benchmarks, such as
    :meth:`env.make_benchmark() <compiler_gym.envs.LlvmEnv.make_benchmark>` for
    LLVM.

    The data underlying a Benchmark instance should be considered immutable. New
    attributes cannote be assigned to Benchmark instances.

    Benchmarks may provide additional functionality such as runtime checks or
    methods for validating the semantics of a benchmark. The benchmark for an
    environment can be set during :meth:`env.reset()
    <compiler_gym.envs.CompilerEnv.reset>`. The currently active benchmark can
    be queried using :attr:`env.benchmark
    <compiler_gym.envs.CompilerEnv.benchmark>`:

        >>> env = gym.make("llvm-v0")
        >>> env.reset(benchmark="cbench-v1/crc32")
        >>> env.benchmark
        cbench-v1/crc32

    A Benchmark instance wraps an instance of the :code:`Benchmark` protocol
    buffer from the `RPC interface
    <https://github.com/facebookresearch/CompilerGym/blob/development/compiler_gym/service/proto/compiler_gym_service.proto>`_
    with additional functionality.
    """

    __slots__ = ["_proto", "_validation_callbacks"]

    def __init__(
        self,
        proto: BenchmarkProto,
        validation_callbacks: Optional[List[ValidationCallback]] = None,
    ):
        self._proto = proto
        self._validation_callbacks = validation_callbacks or []

    def __repr__(self) -> str:
        return str(self.uri)

    @property
    def uri(self) -> str:
        """The URI of the benchmark.

        :return: A URI string.
        :type: string
        """
        return self._proto.uri

    @property
    def proto(self) -> BenchmarkProto:
        """The protocol buffer representing the benchmark.

        :return: A Benchmark message.
        :type: :code:`Benchmark`
        """
        return self._proto

    def is_validatable(self) -> bool:
        """Whether the benchmark has any validation callbacks registered.

        :return: :code:`True` if the benchmark has at least one validation
            callback.
        """
        return self._validation_callbacks != []

    def validate(self, env: "CompilerEnv") -> List[ValidationError]:  # noqa: F821
        """Run any validation callbacks and return any errors.

        If no errors are returned, validation has succeeded:

            >>> benchmark.validate(env)
            []

        If an error occurs, a :class:`ValidationError
        <compiler_gym.ValidationError>` tuple will describe the type of the
        error, and optionally contain other data:

            >>> benchmark.validate(env)
            [ValidationError(type="RuntimeError")]

        Multiple :class:`ValidationError <compiler_gym.ValidationError>` errors
        may be returned to indicate multiple errors.

        This is a synchronous version of :meth:`ivalidate()
        <compiler_gym.datasets.Benchmark.ivalidate>` that blocks until all
        results are ready:

            >>> benchmark.validate(env) == list(benchmark.ivalidate(env))
            True

        :param env: The :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
            instance that is being validated.

        :return: A list of zero or more :class:`ValidationError
            <compiler_gym.ValidationError>` tuples that occurred during
            validation.
        """
        return list(self.ivalidate(env))

    def ivalidate(self, env: "CompilerEnv") -> Iterable[ValidationError]:  # noqa: F821
        """Run any validation callbacks and return a generator of errors.

        This is an asynchronous version of :meth:`validate()
        <compiler_gym.datasets.Benchmark.validate>` that returns immediately.

        :parameter env: A :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`
            instance to validate.

        :return: A generator of :class:`ValidationError
            <compiler_gym.ValidationError>` tuples that occur during validation.
        """
        executor = thread_pool.get_thread_pool_executor()
        futures = (
            executor.submit(validator, env) for validator in self.validation_callbacks()
        )
        for future in as_completed(futures):
            result: Iterable[ValidationError] = future.result()
            if result:
                yield from result

    def validation_callbacks(
        self,
    ) -> List[ValidationCallback]:
        """Return the list of registered validation callbacks.

        :return: A list of callables. See :meth:`add_validation_callback()
            <compiler_gym.datasets.Benchmark.add_validation_callback>`.
        """
        return self._validation_callbacks

    def add_validation_callback(
        self,
        validation_callback: ValidationCallback,
    ) -> None:
        """Register a new validation callback that will be executed on
        :meth:`validate() <compiler_gym.datasets.Benchmark.validate>`.

        :param validation_callback: A callback that accepts a single
            :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` argument and
            returns an iterable sequence of zero or more :class:`ValidationError
            <compiler_gym.ValidationError>` tuples. Validation callbacks must be
            thread safe and must not modify the environment.
        """
        self._validation_callbacks.append(validation_callback)

    @classmethod
    def from_file(cls, uri: str, path: Path):
        """Construct a benchmark from the path to a file.

        :param uri: The URI of the benchmark.

        :param path: A filesystem path.

        :return: A :class:`Benchmark <compiler_gym.datasets.Benchmark>` instance.
        """
        return cls(
            proto=BenchmarkProto(
                uri=uri, program=File(uri=f"file:///{Path(path).absolute()}")
            ),
        )

    @classmethod
    def from_file_contents(cls, uri: str, data: bytes):
        """Construct a benchmark from a raw data array.

        :param uri: The URI of the benchmark.

        :param data: An array of bytes that will be passed to the compiler
            service.
        """
        return cls(proto=BenchmarkProto(uri=uri, program=File(contents=data)))

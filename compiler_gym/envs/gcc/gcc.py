#! /usr/bin/env python3
#
#  Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Query a GCC binary for version,  optimization and param spaces.
The goal of this file is to query the available settings in a GCC compiler so
that they don't have to be hard coded.

The main entry point to this file is the 'get_spec' function which returns a
GccSpec object. That object describes the version, options and parameters.

Querying these settings is time consuming, so this file tries to cache the
values in a cache directory.

Running this file will print the gcc spec to stdout.
"""
import logging
import math
import os
import pickle
import re
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union

import docker

from compiler_gym.service import EnvironmentNotSupported, ServiceError, ServiceInitError
from compiler_gym.util.filesystem import atomic_file_write
from compiler_gym.util.runfiles_path import site_data_path

logger = logging.getLogger(__name__)


class Option:
    """An Option is either a command line optimization setting or a parameter.

    It is essentially a list of the possible values that can be taken. Each item
    is command line parameter. In GCC, all of these are single settings, so only
    need one string to describe them, rather than a list.
    """

    def __len__(self):
        """Number of available settings. Note that the absence of a value is not
        included in this, it is implicit.
        """
        raise NotImplementedError()

    def __getitem__(self, key: int) -> str:
        """Get the command line argument associated with an index (key)."""
        raise NotImplementedError()

    def __str__(self) -> str:
        """Get the name of this option."""
        raise NotImplementedError()


class GccOOption(Option):
    """This class represents the :code:`-O0`, :code:`-O1`, :code:`-O2`,
    :code:`-O3`, :code:`-Os`, and :code:`-Ofast` options.

    This class starts with no values, we fill them in with
    :code:`_gcc_parse_optimize()`. The suffixes to append to :code:`-O` are
    stored in self.values.
    """

    def __init__(self):
        self.values = []

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key: int) -> str:
        return "-O" + self.values[key]

    def __str__(self) -> str:
        return "-O"

    def __repr__(self) -> str:
        return f"<GccOOption values=[{','.join(self.values)}]>"


class GccFlagOption(Option):
    """An ordinary :code:`-f` flag.

    These have two possible settings. For a given flag name there are
    :code:`'-f<name>' and :code:`'-fno-<name>. If :code:`no_fno` is true, then
    there is only the :code:`-f<name>` form.
    """

    def __init__(self, name: str, no_fno: bool = False):
        self.name = name
        self.no_fno = no_fno

    def __len__(self):
        return 1 if self.no_fno else 2

    def __getitem__(self, key: int) -> str:
        return f"-f{'' if key == 0 else 'no-'}{self.name}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagOption name={self.name}>"


class GccFlagEnumOption(Option):
    """A flag of style :code:`-f<name>=[val1, val2, ...]`.

    :code:`self.name` holds the name. :code:`self.values` holds the values.
    """

    def __init__(self, name: str, values: List[str]):
        self.name = name
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key: int) -> str:
        return f"-f{self.name}={self.values[key]}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagEnumOption name={self.name}, values=[{','.join(self.values)}]>"


class GccFlagIntOption(Option):
    """A flag of style :code:`-f<name>=<integer>` where the integer is between
    min and max.
    """

    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max

    def __len__(self):
        return self.max - self.min + 1

    def __getitem__(self, key: int) -> str:
        return f"-f{self.name}={self.min + key}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagIntOption name={self.name}, min={self.min}, max={self.max}>"


class GccFlagAlignOption(Option):
    """Alignment flags. These take several forms. See the GCC documentation."""

    def __init__(self, name: str):
        logger.warning("Alignment options not properly handled %s", name)
        self.name = name

    def __len__(self):
        return 1

    def __getitem__(self, key: int) -> str:
        return f"-f{self.name}"

    def __str__(self) -> str:
        return f"-f{self.name}"

    def __repr__(self) -> str:
        return f"<GccFlagAlignOption name={self.name}>"


class GccParamEnumOption(Option):
    """A parameter :code:`--param=<name>=[val1, val2, val3]`."""

    def __init__(self, name: str, values: List[str]):
        self.name = name
        self.values = values

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key: int) -> str:
        return f"--param={self.name}={self.values[key]}"

    def __str__(self) -> str:
        return f"--param={self.name}"

    def __repr__(self) -> str:
        return (
            f"<GccParamEnumOption name={self.name}, values=[{','.join(self.values)}]>"
        )


class GccParamIntOption(Option):
    """A parameter :code:`--param=<name>=<integer>`, where the integer is
    between min and max.
    """

    def __init__(self, name: str, min: int, max: int):
        self.name = name
        self.min = min
        self.max = max

    def __len__(self):
        return self.max - self.min + 1

    def __getitem__(self, key: int) -> str:
        return f"--param={self.name}={self.min + key}"

    def __str__(self) -> str:
        return f"--param={self.name}"

    def __repr__(self) -> str:
        return f"<GccParamIntOption name={self.name}, min={self.min}, max={self.max}>"


@lru_cache(maxsize=2)
def get_docker_client():
    """Fetch the docker client singleton."""
    try:
        return docker.from_env()
    except docker.errors.DockerException as e:
        raise EnvironmentNotSupported(
            f"Failed to initialize docker client needed by GCC environment: {e}.\n"
            "Have you installed the runtime dependencies?\n See "
            "<https://facebookresearch.github.io/CompilerGym/envs/gcc.html#installation> "
            "for details."
        ) from e


# We only need to run this function once per image.
@lru_cache(maxsize=64)
def pull_docker_image(image: str) -> str:
    """Pull the requested docker image.

    :param image: The name of the docker image to pull.

    :raises ServiceInitError: If pulling the docker image fails.
    """
    try:
        client = get_docker_client()
        client.images.pull(image)
        return image
    except docker.errors.DockerException as e:
        raise ServiceInitError(f"Failed to fetch docker image '{image}': {e}")


def join_docker_container(container, timeout_seconds: int) -> str:
    """Block until the container terminates, returning its output."""
    try:
        status = container.wait(timeout=timeout_seconds)
    except docker.exceptions.ReadTimeout as e:
        # Catch and re-raise the timeout.
        raise TimeoutError(f"GCC timed out after {timeout_seconds:,d} seconds") from e

    if status["StatusCode"]:
        logs = ""
        try:
            logs = container.logs(stdout=True, stderr=False).decode()
        except (UnicodeDecodeError, docker.errors.NotFound):
            pass
        raise ServiceError(f"GCC failed with returncode {status['StatusCode']}: {logs}")

    return container.logs(stdout=True, stderr=False).decode()


class Gcc:
    """This class represents an instance of the GCC compiler, either as a binary
    or a docker image.

    :ivar bin: A string version of the constructor argument.

    :vartype bin: str

    :ivar spec: A :class:`GccSpec <compiler_gym.envs.gcc.gcc.GccSpec>` instance.

    :vartype spec: GccSpec
    """

    def __init__(self, bin: Union[str, Path]):
        self.bin = str(bin)
        self.image = self.bin[len("docker:") :]

        if self.bin.startswith("docker:"):
            pull_docker_image(self.image)
            self.call = self._docker_run
        else:
            self.call = self._subprocess_run

        self.spec = _get_spec(self, cache_dir=site_data_path("gcc-v0"))

    def __call__(
        self,
        *args: str,
        timeout: int,
        cwd: Optional[Path] = None,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> str:
        """Run GCC with the given args.

        :param args: The command line arguments to append.

        :param timeout: A timeout in seconds.

        :param cwd: The working directory.

        :param volumes: A dictionary of volume bindings for docker.

        :raises TimeoutError: If GCC fails to complete within timeout.

        :raises ServiceError: In case GCC fails.
        """
        return self.call(args, timeout, cwd=Path(cwd or "."), volumes=volumes)

    def _docker_run(
        self,
        args: List[str],
        timeout: int,
        cwd: Path,
        volumes: Optional[Dict[str, Dict[str, str]]] = None,
    ):
        cwd = cwd.absolute().as_posix()

        cmd_line = ["gcc"] + list(map(str, args))

        if timeout:
            cmd_line = ["timeout", str(timeout)] + cmd_line

        volumes_ = {cwd: {"bind": cwd, "mode": "rw"}}
        volumes_.update(volumes or {})

        client = get_docker_client()
        container = client.containers.create(
            self.image,
            cmd_line,
            working_dir=cwd,
            volumes=volumes_,
        )
        container.start()
        try:
            return join_docker_container(container, timeout_seconds=timeout)
        finally:
            container.remove()

    def _subprocess_run(self, args, timeout, cwd, volumes):
        del volumes  # Unused

        cmd_line = [self.bin] + list(map(str, args))
        try:
            result = subprocess.check_output(
                cmd_line, cwd=cwd, universal_newlines=True, timeout=timeout
            )
        except subprocess.CalledProcessError as e:
            raise ServiceError(f"Failed to run {self.bin}: {e}") from e
        except FileNotFoundError:
            raise ServiceInitError(f"GCC binary not found: {self.bin}")

        return result


class GccSpec(NamedTuple):
    """This class combines all of the information about the version and options
    for a GCC instance.
    """

    gcc: Gcc
    """A compiler instance."""

    version: str
    """The GCC version string."""

    options: List[Option]
    """A list of options exposed by the compiler."""

    @property
    def size(self) -> int:
        """Calculate the size of the option space. This is the product of the
        cardinalities of all the options.
        """
        sz = 1
        for option in self.options:
            # Each option can be applied or not.
            sz *= len(option) + 1
        return sz


def _gcc_parse_optimize(gcc: Gcc) -> List[Option]:
    """Parse the optimization help string from the GCC binary to find options."""

    logger.debug("Parsing GCC optimization space")

    # Call 'gcc --help=optimize -Q'
    result = gcc("--help=optimize", "-Q", timeout=60)
    # Split into lines. Ignore the first line.
    out = result.split("\n")[1:]

    # Regex patterns to match the different options
    O_num_pat = re.compile("-O<number>")
    O_pat = re.compile("-O([a-z]+)")
    flag_align_eq_pat = re.compile("-f(align-[-a-z]+)=")
    flag_pat = re.compile("-f([-a-z0-9]+)")
    flag_enum_pat = re.compile("-f([-a-z0-9]+)=\\[([-A-Za-z_\\|]+)\\]")
    flag_interval_pat = re.compile("-f([-a-z0-9]+)=<([0-9]+),([0-9]+)>")
    flag_number_pat = re.compile("-f([-a-z0-9]+)=<number>")

    # The list of options as it gets built up.
    options = {}

    # Add a -O value
    def add_gcc_o(value: str):
        # -O flag
        name = "O"
        # There are multiple -O flags. We add one value at a time.
        opt = options[name] = options.get(name, GccOOption())
        # There shouldn't be any way to overwrite this with the wrong type.
        assert type(opt) == GccOOption
        opt.values.append(value)

    # Add a flag
    def add_gcc_flag(name: str):
        # Straight flag.
        # If there is something else in its place already (like a flag enum),
        # then we don't overwrite it.  Straight flags always have the lowest
        # priority
        options[name] = options.get(name, GccFlagOption(name))

    # Add an enum flag
    def add_gcc_flag_enum(name: str, values: List[str]):
        # Enum flag.
        opt = options.get(name)
        if opt:
            # We should only ever be overwriting a straight flag
            assert type(opt) == GccFlagOption
        # Always overwrite
        options[name] = GccFlagEnumOption(name, values)

    # Add an integer flag
    def add_gcc_flag_int(name: str, min: int, max: int):
        # Int flag.
        opt = options.get(name)
        if opt:
            # We should only ever be overwriting a straight flag
            assert type(opt) == GccFlagOption
        # Always overwrite
        options[name] = GccFlagIntOption(name, min, max)

    # Add an align flag
    def add_gcc_flag_align(name: str):
        # Align flag.
        opt = options.get(name)
        if opt:
            # We should only ever be overwriting a straight flag
            assert type(opt) == GccFlagOption
        # Always overwrite
        options[name] = GccFlagAlignOption(name)

    # Parse a line from the help output
    def parse_line(line: str):
        # The first bit of the line is the specification
        bits = line.split()
        if not bits:
            return
        spec = bits[0]

        # -O<number>
        m = O_num_pat.fullmatch(spec)
        if m:
            for i in range(4):
                add_gcc_o(str(i))
            return

        # -Ostr
        m = O_pat.fullmatch(spec)
        if m:
            add_gcc_o(m.group(1))
            return

        # -falign-str=
        # These have quite complicated semantics
        m = flag_align_eq_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            add_gcc_flag_align(name)
            return

        # -fflag
        m = flag_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            add_gcc_flag(name)
            return

        # -fflag=[a|b]
        m = flag_enum_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            values = m.group(2).split("|")
            add_gcc_flag_enum(name, values)
            return

        # -fflag=<min,max>
        m = flag_interval_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = int(m.group(2))
            max = int(m.group(3))
            add_gcc_flag_int(name, min, max)
            return

        # -fflag=<number>
        m = flag_number_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = 0
            max = 2 << 31 - 1
            add_gcc_flag_int(name, min, max)
            return

        logger.warning("Unknown GCC optimization flag spec, '%s'", line)

    # Parse all the lines
    for line in out:
        parse_line(line.strip())

    # Sort and return
    return list(map(lambda x: x[1], sorted(list(options.items()))))


def _gcc_parse_params(gcc: Gcc) -> List[Option]:
    """Parse the param help string from the GCC binary to find options."""

    # Pretty much identical to _gcc_parse_optimize
    logger.debug("Parsing GCC param space")

    result = gcc("--help=param", "-Q", timeout=60)
    out = result.split("\n")[1:]

    param_enum_pat = re.compile("--param=([-a-zA-Z0-9]+)=\\[([-A-Za-z_\\|]+)\\]")
    param_interval_pat = re.compile("--param=([-a-zA-Z0-9]+)=<(-?[0-9]+),([0-9]+)>")
    param_number_pat = re.compile("--param=([-a-zA-Z0-9]+)=")
    param_old_interval_pat = re.compile(
        "([-a-zA-Z0-9]+)\\s+default\\s+(-?\\d+)\\s+minimum\\s+(-?\\d+)\\s+maximum\\s+(-?\\d+)"
    )

    params = {}

    def add_gcc_param_enum(name: str, values: List[str]):
        # Enum param.
        opt = params.get(name)
        assert not opt
        params[name] = GccParamEnumOption(name, values)

    def add_gcc_param_int(name: str, min: int, max: int):
        # Int flag.
        opt = params.get(name)
        assert not opt
        params[name] = GccParamIntOption(name, min, max)

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    def parse_line(line: str):
        bits = line.split()
        if not bits:
            return

        # TODO(hugh): Not sure what the correct behavior is there.
        if len(bits) <= 1:
            return

        spec = bits[0]
        default = bits[1]

        # --param=name=[a|b]
        m = param_enum_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            values = m.group(2).split("|")
            assert not default or default in values
            add_gcc_param_enum(name, values)
            return

        # --param=name=<min,max>
        m = param_interval_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = int(m.group(2))
            max = int(m.group(3))
            if is_int(default):
                assert not default or min <= int(default) <= max
                add_gcc_param_int(name, min, max)
                return

        # --param=name=
        m = param_number_pat.fullmatch(spec)
        if m:
            name = m.group(1)
            min = 0
            max = 2 << 31 - 1
            if is_int(default):
                dflt = int(default)
                min = min if dflt >= min else dflt
                add_gcc_param_int(name, min, max)
                return

        # name  default num minimum num maximum num
        m = param_old_interval_pat.fullmatch(line)
        if m:
            name = m.group(1)
            default = int(m.group(2))
            min = int(m.group(3))
            max = int(m.group(4))
            if min <= default <= max:
                # For now we will only consider fully described params
                add_gcc_param_int(name, min, max)
                return

        logger.warning("Unknown GCC param flag spec, '%s'", line)

    # breakpoint()
    for line in out:
        parse_line(line.strip())

    return list(map(lambda x: x[1], sorted(list(params.items()))))


def _fix_options(options: List[Option]) -> List[Option]:
    """Fixes for things that seem not to be true in the help."""

    def keep(option: Option) -> bool:
        # Ignore -flive-patching
        if isinstance(option, GccFlagEnumOption):
            if option.name == "live-patching":
                return False
        return True

    options = [opt for opt in options if keep(opt)]

    for i, option in enumerate(options):
        if isinstance(option, GccParamIntOption):
            # Some things say they can have -1, but can't
            if option.name in [
                "logical-op-non-short-circuit",
                "prefetch-minimum-stride",
                "sched-autopref-queue-depth",
                "vect-max-peeling-for-alignment",
            ]:
                option.min = 0

        elif isinstance(option, GccFlagOption):
            # -fhandle-exceptions renamed to -fexceptions
            if option.name == "handle-exceptions":
                option.name = "exceptions"

            # Some flags have no -fno- version
            if option.name in [
                "stack-protector-all",
                "stack-protector-explicit",
                "stack-protector-strong",
            ]:
                option.no_fno = True

            # -fno-threadsafe-statics should have the no- removed
            if option.name == "no-threadsafe-statics":
                option.name = "threadsafe-statics"

        elif isinstance(option, GccFlagIntOption):
            # -fpack-struct has to be a small positive power of two
            if option.name == "pack-struct":
                values = [str(1 << j) for j in range(5)]
                options[i] = GccFlagEnumOption("pack-struct", values)

    return options


def _gcc_get_version(gcc: Gcc) -> str:
    """Get the version string"""
    logger.debug("Getting GCC version for %s", gcc.bin)
    try:
        result = gcc("--version", timeout=60)
    except ServiceError as e:
        raise EnvironmentNotSupported(f"Failed to run GCC binary: {gcc.bin}") from e
    version = result.split("\n")[0]
    logger.debug("GCC version is %s", version)
    if "gcc" not in version:
        raise ServiceInitError(f"Invalid GCC version string: {version}")
    return version


def _version_hash(version: str) -> str:
    """Hash the version so we can cache the spec at that name."""
    h = 0
    for c in version:
        h = ord(c) + 31 * h
    return str(h % (2 << 64))


def _get_spec(gcc: Gcc, cache_dir: Path) -> Optional[GccSpec]:
    """Get the specification for a GCC executable.

    :param gcc: The executable.

    :param cache_dir: An optional directory to search for cached versions of the
        spec.
    """
    # Get the version
    version = _gcc_get_version(gcc)

    spec = None
    # See if there is a pickled spec in the cache_dir. First we use a hash to
    # name the file.
    spec_path = cache_dir / _version_hash(version) / "spec.pkl"

    # Try to get the pickled version
    if os.path.isfile(spec_path):
        try:
            with open(spec_path, "rb") as f:
                spec = pickle.load(f)
            spec = GccSpec(gcc=gcc, version=spec.version, options=spec.options)
            logger.debug("GccSpec for version '%s' read from %s", version, spec_path)
        except (pickle.UnpicklingError, EOFError) as e:
            logger.warning("Unable to read spec from '%s': %s", spec_path, e)

    if spec is None:
        # Pickle doesn't exist, parse
        optim_opts = _gcc_parse_optimize(gcc)
        param_opts = _gcc_parse_params(gcc)
        options = _fix_options(optim_opts + param_opts)
        spec = GccSpec(gcc, version, options)
        if not spec.options:
            return None

        # Cache the spec file for future.
        spec_path.parent.mkdir(exist_ok=True, parents=True)
        with atomic_file_write(spec_path, fileobj=True) as f:
            pickle.dump(spec, f)
        logger.debug("GccSpec for %s written to %s", version, spec_path)

    logger.debug("GccSpec size is approximately 10^%.0f", round(math.log(spec.size)))

    return spec

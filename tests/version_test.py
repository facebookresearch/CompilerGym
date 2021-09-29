# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pkg_resources

import compiler_gym
from compiler_gym.util.runfiles_path import runfiles_path
from packaging import version
from tests.pytest_plugins.common import bazel_only, install_test_only
from tests.test_main import main


def test_version_dunder():
    assert isinstance(compiler_gym.__version__, str)


def test_version_dunder_format():
    version.parse(compiler_gym.__version__)


@install_test_only
def test_setuptools_version():
    version = pkg_resources.require("compiler_gym")[0].version
    assert version == compiler_gym.__version__


@bazel_only
def test_expected_version():
    """Test that embedded compiler gym version matches VERSION file."""
    with open(runfiles_path("VERSION")) as f:
        version = f.read().strip()
    assert version == compiler_gym.__version__


if __name__ == "__main__":
    main()

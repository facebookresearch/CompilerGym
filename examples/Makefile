# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
define HELP
CompilerGym usage examples. See README.md for details. Available targets:

    make install
        Install the example python packages.

    make test
        Run the example unit tests.
endef
export HELP

PYTHON ?= python3

install:
	$(PYTHON) setup.py install

test:
	$(PYTHON) -m pytest .

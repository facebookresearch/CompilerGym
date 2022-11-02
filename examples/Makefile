# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
define HELP
CompilerGym usage examples. See README.md for details. Available targets:

    make install
        Install the example python packages.

    make test
        Run pytest in this directory.

    make clean
        Remove build artifacts.

    make uninstall
        Uninstall the example python packages.
endef
export HELP

# The path of the repository reoot.
ROOT := $(realpath $(dir $(realpath $(firstword $(MAKEFILE_LIST))))/..)

PYTHON ?= python3

# Extra command line arguments for pytest.
PYTEST_ARGS ?=

# The path of the XML pytest coverage report to generate when running the
# test-cov target.
COV_REPORT ?= $(ROOT)/coverage.xml

DISTTOOLS_OUTS := \
	build \
	compiler_gym_examples.egg-info \
	dist \
	$(NULL)

.DEFAULT_GOAL := help

help:
	@echo "$$HELP"

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) setup.py install

uninstall:
	$(PYTHON) -m pip uninstall -y compiler_gym_examples

# DGL creates ~/.dgl on first run and I have found that this will fail if tests
# are invoked before this file exists. Suppress errors.
dgl-init:
	python3 -c 'import dgl; print(dgl.__version__)' &>/dev/null || true

# The target to use. This is a relative file path of the directory or file to
# test. If not provided, all tests will be run
TEST_TARGET ?= .

test: dgl-init
	pytest --no-success-flaky-report --benchmark-disable -n auto --durations=5 $(TEST_TARGET) --cov=compiler_gym --cov-report=xml:$(COV_REPORT) $(PYTEST_ARGS)

clean:
	rm -rf $(DISTTOOLS_OUTS)

disclean: clean

purge: distclean uninstall

.PHONY: help install uninstall dgl-init test clean distclean purge

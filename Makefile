define HELP
CompilerGym $(VERSION). Available targets:

Setting up
----------

    make init
        Install the build and runtime python dependencies. This should be run
        once before any other targets.


Testing
-------

    make test
        Run the test suite. Test results are cached so that incremental test
        runs are minimal and fast. Use this as your go-to target for testing
        modifications to the codebase.

    make itest
        Run the test suite continuously on change. This is equivalent to
        manually running `make test` when a source file is modified. Note that
        `make install-test` tests are not run. This requires bazel-watcher.
        See: https://github.com/bazelbuild/bazel-watcher#installation


Post-installation Tests
-----------------------

    make install-test
        Run the full test suite against an installed CompilerGym package. This
        requires that the CompilerGym package has been installed (`make
        install`). This is useful for checking the package contents but is
        usually not needed for interactive development since `make test` runs
        the same tests without having to install anything.

    make install-test-cov
        The same as `make install-test`, but with python test coverage
        reporting. A summary of test coverage is printed at the end of execution
        and the full details are recorded in a coverage.xml file in the project
        root directory. To print a report of file coverage to stdout at the end
        of testing, use argument `PYTEST_ARGS="--cov-report=term"`.

    make install-fuzz
        Run the fuzz testing suite against an installed CompilerGym package.
        Fuzz tests are tests that generate their own inputs and run in a loop
        until an error has been found, or until a minimum number of seconds have
        elapsed. This minimum time is controlled using a FUZZ_SECONDS variable.
        The default is 300 seconds (5 minutes). Override this value at the
        command line, for example `FUZZ_SECONDS=60 make install-fuzz` will run
        the fuzz tests for a minimum of one minute. This requires that the
        CompilerGym package has been installed (`make install`).

    make examples-test
        Run pytest in the examples directory. This requires that the CompilerGym
        package has been installed (`make install`).


Documentation
-------------

    make docs
        Build the HTML documentation using Sphinx. This is the documentation
        site that is hosted at <https://facebookresearch.github.io/CompilerGym>.
        The generated HTML files are in docs/build/html.

    make livedocs
        Build the HTML documentation and serve them on localhost:8000. Changes
        to the documentation will automatically trigger incremental rebuilds
        and reload the changes.  To change the host and port, set the SPHINXOPTS
        env variable: `SPHINXOPTS="--port 1234 --host 0.0.0.0" make livedocs`


Deployment
----------

    make bdist_wheel
        Build an optimized python wheel. The generated file is in
        dist/compiler_gym-<version>-<platform_tags>.whl

    make install
        Build and install the python wheel.

    make bdist_wheel-linux
        Use a docker container to build a python wheel for linux. This is only
        used for making release builds. This requires docker.

    make bdist_wheel-docker
        Build a docker image containing CompilerGym.

    bdist_wheel-linux-shell
        Drop into a bash terminal in the docker container that is used for
        linux builds. This may be useful for debugging bdist_wheel-linux
        builds.

    make bdist_wheel-linux-test
        Run the `make install-test` suite against the build artifact generated
        by `make bdist_wheel-linux`.

	make www
		Run a local instance of the web visualization service. See www/README.md
		for details.

Tidying up
-----------

    make clean
        Remove build artifacts.

    make distclean
        Clean up all build artifacts, including the build cache.

    make uninstall
        Uninstall the python package.

    make purge
        Uninstall the python package and completely remove all datasets, logs,
        and cached files. Any experimental data or generated logs will be
        irreversibly deleted!
endef
export HELP

# Configurable paths to binaries.
CC ?= clang
CXX ?= clang++
BAZEL ?= bazel
DOXYGEN ?= doxygen
IBAZEL ?= ibazel
PANDOC ?= pandoc
PYTHON ?= python3

# Bazel build options.
BAZEL_OPTS ?=
BAZEL_FETCH_OPTS ?=
BAZEL_BUILD_OPTS ?= -c opt
BAZEL_TEST_OPTS ?=

# The path of the repository reoot.
ROOT := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))

VERSION := $(shell cat VERSION)
OS := $(shell uname)


##############
# Setting up #
##############

.DEFAULT_GOAL := help

.PHONY: help init

help:
	@echo "$$HELP"

init:
	$(PYTHON) -m pip install -r requirements.txt
	pre-commit install

init-runtime-requirements:
	$(PYTHON) -m pip install -r compiler_gym/requirements.txt

############
# Building #
############

# Files and directories generated by python disttools.
DISTTOOLS_OUTS := \
	build \
	compiler_gym.egg-info \
	dist \
	examples/build \
	examples/compiler_gym_examples.egg-info \
	examples/dist \
	$(NULL)

BUILD_TARGET ?= //:package

BAZEL_FETCH_RETRIES ?= 5

# Run `bazel fetch` in a retry loop due to intermitent failures when fetching
# remote archives in the CI environment.
bazel-fetch:
	@for i in $$(seq 1 $(BAZEL_FETCH_RETRIES)); do \
		echo "$(BAZEL) $(BAZEL_OPTS) fetch $(BAZEL_FETCH_OPTS) $(BUILD_TARGET)"; \
		if $(BAZEL) $(BAZEL_OPTS) fetch $(BAZEL_FETCH_OPTS) $(BUILD_TARGET) ; then \
			break; \
		else \
			echo "bazel fetch attempt $$i of $(BAZEL_FETCH_RETRIES) failed" >&2; \
		fi; \
		if [ $$i -eq 10 ]; then \
			false; \
		fi; \
	done

bazel-build: bazel-fetch
	$(BAZEL) $(BAZEL_OPTS) build $(BAZEL_BUILD_OPTS) $(BUILD_TARGET)

bdist_wheel: bazel-build
	$(PYTHON) setup.py bdist_wheel

bdist_wheel-linux-rename:
	mv dist/compiler_gym-$(VERSION)-py3-none-linux_x86_64.whl dist/compiler_gym-$(VERSION)-py3-none-manylinux2014_x86_64.whl

# The docker image to use for building the bdist_wheel-linux target. See
# packaging/Dockerfile.
MANYLINUX_DOCKER_IMAGE ?= chriscummins/compiler_gym-manylinux-build:2021-09-21

bdist_wheel-linux:
	rm -rf build
	docker pull $(MANYLINUX_DOCKER_IMAGE)
	docker run -v $(ROOT):/CompilerGym --workdir /CompilerGym --rm --shm-size=8g "$(MANYLINUX_DOCKER_IMAGE)" /bin/sh -c './packaging/compiler_gym-manylinux-build/container_init.sh && make bdist_wheel bdist_wheel-linux-rename BAZEL_OPTS="$(BAZEL_OPTS)" BAZEL_BUILD_OPTS="$(BAZEL_BUILD_OPTS)" BAZEL_FETCH_OPTS="$(BAZEL_FETCH_OPTS)" && rm -rf build'

bdist_wheel-linux-shell:
	docker run -v $(ROOT):/CompilerGym --workdir /CompilerGym --rm --shm-size=8g -it --entrypoint "/bin/bash" "$(MANYLINUX_DOCKER_IMAGE)"

bdist_wheel-linux-test:
	docker run -v $(ROOT):/CompilerGym --workdir /CompilerGym --rm --shm-size=8g "$(MANYLINUX_DOCKER_IMAGE)" /bin/sh -c 'cd /CompilerGym && pip3 install -U pip && pip3 install dist/compiler_gym-$(VERSION)-py3-none-manylinux2014_x86_64.whl && pip install -r tests/requirements.txt && make install-test'

bdist_wheel-docker: bdist_wheel-linux
	cp dist/compiler_gym-$(VERSION)-py3-none-manylinux2014_x86_64.whl packaging/compiler_gym-local-wheel
	docker build -t chriscummins/compiler_gym:latest packaging/compiler_gym-local-wheel
	docker build -t chriscummins/compiler_gym:$(VERSION) packaging/compiler_gym-local-wheel

all: docs bdist_wheel bdist_wheel-linux

.PHONY: bazel-fetch bazel-build bdist_wheel bdist_wheel-linux bdist_wheel-linux-shell bdist_wheel-linux-test

#################
# Web interface #
#################

# A list of in-tree files generated by the www project build.
WWW_OUTS = \
	www/frontends/compiler_gym/build \
	www/frontends/compiler_gym/node_modules \
	$(NULL)

# The name of the docker image built by the "www-image" target.
WWW_IMAGE_TAG ?= chriscummins/compiler_gym-www

www: www-build
	cd www && $(PYTHON) www.py

www-build:
	cd www/frontends/compiler_gym && npm ci && npm run build

www-image: www-build
	cd www && docker build -t "$(WWW_IMAGE_TAG)" .
	docker run -p 5000:5000 "$(WWW_IMAGE_TAG)"

.PHONY: www www-build

#################
# Documentation #
#################

docs/source/changelog.rst: CHANGELOG.md
	echo "..\n  Generated from $<. Do not edit!\n" > $@
	echo "Changelog\n=========\n" >> $@
	$(PANDOC) --from=markdown --to=rst $< >> $@

docs/source/contributing.rst: CONTRIBUTING.md
	echo "..\n  Generated from $<. Do not edit!\n" > $@
	$(PANDOC) --from=markdown --to=rst $< >> $@

GENERATED_DOCS := \
	docs/source/changelog.rst \
	docs/source/contributing.rst \
	$(NULL)

gendocs: $(GENERATED_DOCS)

doxygen:
	cd docs && $(DOXYGEN) Doxyfile

doxygen-rst:
	cd docs && $(PYTHON) generate_cc_rst.py

docs: gendocs bazel-build doxygen
	PYTHONPATH=$(ROOT)/bazel-bin/package.runfiles/CompilerGym sphinx-build -M html docs/source docs/build $(SPHINXOPTS)

livedocs: gendocs doxygen
	PYTHONPATH=$(ROOT)/bazel-bin/package.runfiles/CompilerGym sphinx-autobuild docs/source docs/build $(SPHINXOPTS) --pre-build 'make gendocs bazel-build doxygen' --watch compiler_gym


.PHONY: doxygen doxygen-rst


###########
# Testing #
###########

COMPILER_GYM_SITE_DATA ?= /tmp/compiler_gym_$(USER)/tests/site_data
COMPILER_GYM_CACHE ?= /tmp/compiler_gym_$(USER)/tests/cache

# A directory that is used as the working directory for running pytest tests
# by symlinking the tests directory into it.
INSTALL_TEST_ROOT ?= /tmp/compiler_gym_$(USER)/install_tests

# The target to use. If not provided, all tests will be run. For `make test` and
# related, this is a bazel target pattern, with default value '//...'. For `make
# install-test` and related, this is a relative file path of the directory or
# file to test, with default value 'tests'.
TEST_TARGET ?=

# Extra command line arguments for pytest.
PYTEST_ARGS ?=

# The path of the XML pytest coverage report to generate when running the
# install-test-cov target.
COV_REPORT ?= $(ROOT)/coverage.xml

test: bazel-fetch
	$(BAZEL) $(BAZEL_OPTS) test $(BAZEL_TEST_OPTS) $(if $(TEST_TARGET),$(TEST_TARGET),//...)

itest: bazel-fetch
	$(IBAZEL) $(BAZEL_OPTS) test $(BAZEL_TEST_OPTS) $(if $(TEST_TARGET),$(TEST_TARGET),//...)

# Since we can't run compiler_gym from the project root we need to jump through
# some hoops to run pytest "out of tree" by creating an empty directory and
# symlinking the test directory into it so that pytest can be invoked.
install-test-setup:
	mkdir -p "$(INSTALL_TEST_ROOT)"
	rm -f "$(INSTALL_TEST_ROOT)/tests" "$(INSTALL_TEST_ROOT)/tox.ini"
	ln -s "$(ROOT)/tests" "$(INSTALL_TEST_ROOT)"
	ln -s "$(ROOT)/tox.ini" "$(INSTALL_TEST_ROOT)"

define pytest
	cd "$(INSTALL_TEST_ROOT)" && pytest $(if $(TEST_TARGET),$(TEST_TARGET),tests) $(1) $(PYTEST_ARGS)
endef

install-test: | install-test-setup
	$(call pytest,--no-success-flaky-report --benchmark-disable -n auto -k "not fuzz" --durations=5)

examples-pip-install:
	cd examples && python setup.py install

examples-test: examples-pip-install
	cd examples && pytest --nbmake --no-success-flaky-report --benchmark-disable -n auto --durations=5 . --cov=compiler_gym --cov-report=xml:$(COV_REPORT) $(PYTEST_ARGS)

# Note we export $CI=1 so that the tests always run as if within the CI
# environement. This is to ensure that the reported coverage matches that of
# the value on: https://codecov.io/gh/facebookresearch/CompilerGym
install-test-cov: install-test-setup
	export CI=1; $(call pytest,--no-success-flaky-report --benchmark-disable -n auto -k "not fuzz" --durations=5 --cov=compiler_gym --cov-report=xml:$(COV_REPORT))

# The minimum number of seconds to run the fuzz tests in a loop for. Override
# this at the commandline, e.g. `FUZZ_SECONDS=1800 make fuzz`.
FUZZ_SECONDS ?= 300

install-fuzz: install-test-setup
	$(call pytest,--no-success-flaky-report -p no:sugar -x -vv -k fuzz --seconds=$(FUZZ_SECONDS))

post-install-test:
	$(MAKE) -C examples/makefile_integration clean
	SEARCH_TIME=3 $(MAKE) -C examples/makefile_integration test

.PHONY: test post-install-test examples-pip-install examples-test


################
# Installation #
################

# We run uninstall as a dependency of install to prevent conflicting
# CompilerGym versions co-existing in the same Python environment.

pip-install: uninstall
	$(PYTHON) setup.py install

install: |  init-runtime-requirements bazel-build pip-install

.PHONY: pip-install install


##############
# Tidying up #
##############

# A list of all filesystem locations that CompilerGym may use for storing
# files and data.
COMPILER_GYM_DATA_FILE_LOCATIONS = \
    "$(HOME)/.cache/compiler_gym" \
    "$(HOME)/.local/share/compiler_gym" \
    "$(HOME)/logs/compiler_gym" \
    /dev/shm/compiler_gym \
    /dev/shm/compiler_gym_$(USER) \
    /tmp/compiler_gym \
    /tmp/compiler_gym_$(USER) \
    $(NULL)

.PHONY: clean distclean uninstall purge

clean:
	rm -rf $(GENERATED_DOCS) $(DISTTOOLS_OUTS) $(WWW_OUTS)
	find . -type d -name __pycache__ -o -name .benchmarks -print0 | xargs -0 -I {} /bin/rm -rf "{}"
	find . -type f -name '.coverage*' -delete

distclean: clean
	bazel clean --expunge

uninstall:
	$(PYTHON) -m pip uninstall -y compiler_gym

purge: distclean uninstall
	rm -rf $(COMPILER_GYM_DATA_FILE_LOCATIONS)

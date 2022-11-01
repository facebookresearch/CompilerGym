#!/usr/bin/env bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Perform post-launch initialization of a docker container for building
# CompilerGym. Usage:
#
#     make bdist_wheel-linux-shell
#     # in docker:
#     bash packaging/container_init.sh
#     make bdist_wheel
set -euxo pipefail

apt-get update
pip3 install --no-cache-dir -U setuptools pip wheel

# Filter grpcio as there are problems with installing it at this stage.
# Note the use of a tempfile to store the filtered requirements rather than
# just `xargs pip install` because we need to use `pip install -r` to parse
# the requirements file syntax.
grep -v '^grpcio' compiler_gym/requirements.txt > /tmp/requirements.txt
pip3 install --no-cache-dir -r /tmp/requirements.txt
rm -f /tmp/requiremts.txt

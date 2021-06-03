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
grep -v grpc compiler_gym/requirements.txt | xargs pip3 install

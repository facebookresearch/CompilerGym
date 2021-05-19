#!/usr/bin/env bash
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

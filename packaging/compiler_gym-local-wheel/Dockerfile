# syntax=docker/dockerfile:1
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

FROM python:3.8-slim-buster

LABEL maintainer="Chris Cummins <cummins@fb.com>"

# The version of CompilerGym to install.
ENV COMPILER_GYM_VERSION=0.1.10
# Put the runtime downloads in a convenient location.
ENV COMPILER_GYM_CACHE=/compiler_gym/cache
ENV COMPILER_GYM_SITE_DATA=/compiler_gym/site_data

# We need a C/C++ toolchain to build the CompilerGym python dependencies and to
# provide the system includes for the LLVM environment.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libtinfo5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create an unversioned library for libtinfo5 so that -ltinfo works.
RUN ln -s /lib/x86_64-linux-gnu/libtinfo.so.5 /lib/x86_64-linux-gnu/libtinfo.so

# Add the CompilerGym wheel and install it.
COPY compiler_gym*.whl /
RUN \
    python3 -m pip install --no-cache-dir compiler_gym*.whl && \
    rm compiler_gym*.whl

# Run the LLVM environment now to download the LLVM runtime requirements, then
# delete the cached downloads.
RUN \
    python3 -m compiler_gym.bin.service --env=llvm-v0 && \
    rm -rf "$COMPILER_GYM_CACHE"

ENTRYPOINT ["python3", "-m", "compiler_gym.bin.service", "--run_on_port=5000"]

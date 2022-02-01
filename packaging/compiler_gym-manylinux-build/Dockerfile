# syntax=docker/dockerfile:1
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# A linux environment for building CompilerGym wheels for Linux.

FROM ubuntu:20.04

LABEL maintainer="Chris Cummins <cummins@fb.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Put the runtime downloads in a convenient location.
ENV COMPILER_GYM_CACHE=/compiler_gym/cache
ENV COMPILER_GYM_SITE_DATA=/compiler_gym/site_data
ENV COMPILER_GYM_TRANSIENT_CACHE=/compiler_gym/cache/transient

# We need a C/C++ toolchain to build the CompilerGym python dependencies and to
# provide the system includes for the LLVM environment.
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        python3-dev \
        python3-distutils \
        python3-pip \
        python3 \
        build-essential \
        libtinfo5 \
        gnupg \
        software-properties-common \
        apt-transport-https \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create an unversioned library for libtinfo5 so that -ltinfo works.
RUN python3 -m pip install --no-cache-dir 'setuptools==50.3.2' 'wheel==0.36.0' \
  && ln -s /lib/x86_64-linux-gnu/libtinfo.so.5 /lib/x86_64-linux-gnu/libtinfo.so

ENV CC=clang-12
ENV CXX=clang++-12

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
# hadolint ignore=DL3008
RUN curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add - \
    && apt-add-repository "deb http://apt.llvm.org/focal/ llvm-toolchain-focal-12 main" \
    && apt-get install -y --no-install-recommends clang-12

# install bazel
# hadolint ignore=DL3015,DL4006
RUN curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor > bazel.gpg \
    && mv bazel.gpg /etc/apt/trusted.gpg.d/ \
    && echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list \
    && apt-get update  \
    && apt-get install -y bazel=4.2.2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /home/NumericsGym

# hadolint ignore=DL3013
RUN pip3 install --no-cache-dir virtualenv
# hadolint ignore=SC1091
RUN virtualenv --python=python3 /home/venv \
    && source /home/venv/bin/activate

# RUN make init
# CMD ["bazel", "build", "..."]
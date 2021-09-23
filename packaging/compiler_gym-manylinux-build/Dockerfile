# syntax=docker/dockerfile:1
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# A linux environment for building CompilerGym wheels for Linux.
#
# CompilerGym builts against LLVM binaries for Ubuntu 18.04. This docker image
# adds the CompilerGym dependencies for building python wheels.
FROM ubuntu:18.04

LABEL maintainer="Chris Cummins <cummins@fb.com>"

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        clang \
        cmake \
        curl \
        libtinfo5 \
        m4 \
        make \
        patch \
        patchelf \
        python3-dev \
        python3-distutils \
        python3-pip \
        python3 \
        rsync \
        zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN curl -L "https://github.com/bazelbuild/bazelisk/releases/download/v1.6.1/bazelisk-linux-amd64" > bazel.tmp && mv bazel.tmp /usr/bin/bazel && chmod +x /usr/bin/bazel

RUN python3 -m pip install --no-cache-dir 'setuptools==50.3.2' 'wheel==0.36.0'

# Building grpc requires python 2.
# Known issue: https://github.com/grpc/grpc/pull/24407
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        'python-dev=2.7.15~rc1-1' \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create an unversioned library for libtinfo5 so that -ltinfo works.
RUN ln -s /lib/x86_64-linux-gnu/libtinfo.so.5 /lib/x86_64-linux-gnu/libtinfo.so

ENV CC=clang
ENV CXX=clang++

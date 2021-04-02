# A linux environment for building CompilerGym wheels for Linux.
#
# CompilerGym builts against LLVM binaries for Ubuntu 18.04. This docker image
# adds the CompilerGym dependencies for building python wheels.
FROM ubuntu:18.04

LABEL maintainer="Chris Cummins <cummins@fb.com>"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        'clang=1:6.0-41~exp5~ubuntu1' \
        'cmake=3.10.2-1ubuntu2.18.04.1' \
        'curl=7.58.0-2ubuntu3.12' \
        'make=4.1-9.1ubuntu1' \
        'patchelf=0.9-1' \
        'python3-dev=3.6.7-1~18.04' \
        'python3-distutils=3.6.9-1~18.04' \
        'python3-pip=9.0.1-2.3~ubuntu1.18.04.4' \
        'python3=3.6.7-1~18.04' \
        'libtinfo5=6.1-1ubuntu1.18.04' \
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

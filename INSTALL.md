# Installation

Install the latest CompilerGym release using:

    pip install -U compiler_gym

CompilerGym requires Python >= 3.6. The binary works on macOS and Linux (on
Ubuntu 18.04, Fedora 28, Debian 10 or newer equivalents).

# Building from Source

If you prefer, you may build from source. This requires a modern C++ toolchain
and bazel.

## macOS

On macOS the required dependencies can be installed using
[homebrew](https://docs.brew.sh/Installation):

```sh
brew install bazelisk buildifier clang-format hadolint prototool zlib
export LDFLAGS="-L/usr/local/opt/zlib/lib"
export CPPFLAGS="-I/usr/local/opt/zlib/include"
export PKG_CONFIG_PATH="/usr/local/opt/zlib/lib/pkgconfig"
```

Now proceed to [All platforms](#all-platforms) below.

## Linux

On debian-based linux systems, install the required toolchain using:

```sh
sudo apt install -y clang-9 clang++-9 clang-format golang libjpeg-dev \
  libtinfo5 m4 make patch zlib1g-dev tar bzip2 wget
mkdir -pv ~/.local/bin
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O ~/.local/bin/bazel
wget https://github.com/hadolint/hadolint/releases/download/v1.19.0/hadolint-Linux-x86_64 -O ~/.local/bin/hadolint
chmod +x ~/.local/bin/bazel ~/.local/bin/hadolint
go get github.com/bazelbuild/buildtools/buildifier
GO111MODULE=on go get github.com/uber/prototool/cmd/prototool@dev
export PATH="$HOME/.local/bin:$PATH"
export CC=clang
export CXX=clang++
```


## All platforms

We recommend using
[conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
to manage the remaining build dependencies. First create a conda environment
with the required dependencies:

    conda create -y -n compiler_gym python=3.8
    conda activate compiler_gym
    conda install -y -c conda-forge cmake pandoc patchelf

Then clone the CompilerGym source code using:

    git clone https://github.com/facebookresearch/CompilerGym.git
    cd CompilerGym

There are two primary git branches: `stable` tracks the latest release;
`development` is for bleeding edge features that may not yet be mature. Checkout
your preferred branch and install the python development dependencies using:

    git checkout stable
    make init

The `make init` target only needs to be run on initial setup and after pulling
remote changes to the CompilerGym repository.

## Building from source with Bazel

Run the test suite to confirm that everything is working:

    make test

To build and install the `compiler_gym` python package, run:

    make install

**NOTE:** To use the `compiler_gym` package that is installed by `make install`
you must leave the root directory of this repository. Attempting to import
`compiler_gym` while in the root of this repository will cause import errors.

When you are finished, you can deactivate and delete the conda
environment using:

    conda deactivate
    conda env remove -n compiler_gym

## Building from source with CMake

Darwin is not supported with CMake.

### Dependency instructions for Ubuntu

```bash
sudo apt-get install g++ lld-9 \
  autoconf libtool ninja-build ccache git \
```

Requires CMake (>=3.20).

```bash
wget https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-x86_64.sh -O cmake.sh
bash cmake.sh --prefix=$HOME/.local --exclude-subdir --skip-license
rm cmake.sh
```

### Dependency Arguments
By default most dependencies are built together with Compiler Gym. To search for a dependency instead use:

```
-DCOMPILER_GYM_<dependency>_PROVIDER=external
```

* `COMPILER_GYM_BOOST_PROVIDER`
* `COMPILER_GYM_GFLAGS_PROVIDER`
* `COMPILER_GYM_GLOG_PROVIDER`
* `COMPILER_GYM_GRPC_PROVIDER`
* `COMPILER_GYM_GTEST_PROVIDER`
* `COMPILER_GYM_NLOHMANN_JSON_PROVIDER`
* `COMPILER_GYM_PROTOBUF_PROVIDER`

```bash
cmake -GNinja \
  -DCMAKE_C_COMPILER=clang-9 \
  -DCMAKE_CXX_COMPILER=clang++-9 \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \ # For faster rebuilds, can be removed
  -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld" -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld" \ # For faster builds, can be removed
  -DPython3_FIND_VIRTUALENV=FIRST \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=true \
  -S "<path to source directory>" \
  -B "<path to build directory>"

cmake  --build "<path to build directory>"

pip install <path to build directory>/py_pkg/dist/compiler_gym*.whl --force-reinstall
```
Additional optional configuration arguments:

* Enables testing.

    ```bash
    -DCOMPILER_GYM_BUILD_TESTS=ON
    ```

* Builds additional tools required by some examples.

    ```bash
    -DCOMPILER_GYM_BUILD_EXAMPLES=ON
    ```

* For faster linking.

    ```bash
    -DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld-9"
    -DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld-9"
    -DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld-9"
    ```

* For faster rebuilds.

    ```bash
    -DCMAKE_C_COMPILER_LAUNCHER=ccache
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    ```

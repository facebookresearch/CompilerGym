# Installation

Install the latest CompilerGym release using:

    pip install -U compiler_gym

CompilerGym requires Python >= 3.7. The binary works on macOS and Linux (on
Ubuntu 18.04, Fedora 28, Debian 10 or newer equivalents).

# Building from Source

If you prefer, you may build from source. This requires a modern C++ toolchain
Bazel and CMake.

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
sudo apt install -y clang clang-format golang libjpeg-dev \
  libtinfo5 m4 make patch zlib1g-dev tar bzip2 wget
mkdir -pv ~/.local/bin
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.7.5/bazelisk-linux-amd64 -O ~/.local/bin/bazel
wget https://github.com/hadolint/hadolint/releases/download/v1.19.0/hadolint-Linux-x86_64 -O ~/.local/bin/hadolint
chmod +x ~/.local/bin/bazel ~/.local/bin/hadolint
go install github.com/bazelbuild/buildtools/buildifier@latest
GO111MODULE=on go install github.com/uber/prototool/cmd/prototool@dev
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
    conda install -y -c conda-forge cmake doxygen pandoc patchelf

Then clone the CompilerGym source code using:

    git clone https://github.com/facebookresearch/CompilerGym.git
    cd CompilerGym

If you plan to contribute to CompilerGym, install the development environment
requirements using:

    make dev-init


### Building from source with Bazel

To build and install the `compiler_gym` python package, run:

    make install

Once this has completed, run the test suite on the installed package using:

    make test

This may take a while. There are a number of options to `make test`, see `make
help` for more information.

Each time you modify the sources it is necessary to rerun `make install` before
`make test`.

**NOTE:** To use the `compiler_gym` package that is installed by `make install`
you must leave the root directory of this repository. Attempting to import
`compiler_gym` while in the root of this repository will cause an import error.

When you are finished, you can deactivate and delete the conda
environment using:

    conda deactivate
    conda env remove -n compiler_gym

### Building from source with CMake

Building with CMake is experimental and supports only Linux.

Install the dependencies using:

```sh
sudo apt-get install g++ lld autoconf libtool ninja-build ccache git
```

Requires CMake (>=3.20).

```sh
wget https://github.com/Kitware/CMake/releases/download/v3.20.5/cmake-3.20.5-linux-x86_64.sh -O cmake.sh
bash cmake.sh --prefix=$HOME/.local --exclude-subdir --skip-license
rm cmake.sh
export PATH=$HOME/.local/bin:$PATH
```

By default most dependencies are built together with Compiler Gym. To search for a dependency instead use:

```
-DCOMPILER_GYM_<dependency>_PROVIDER=external
```

* `COMPILER_GYM_BOOST_PROVIDER`
* `COMPILER_GYM_GFLAGS_PROVIDER`
* `COMPILER_GYM_GLOG_PROVIDER`
* `COMPILER_GYM_GRPC_PROVIDER`
* `COMPILER_GYM_GTEST_PROVIDER`
* `COMPILER_GYM_LLVM_PROVIDER`
* `COMPILER_GYM_NLOHMANN_JSON_PROVIDER`
* `COMPILER_GYM_PROTOBUF_PROVIDER`

```sh
cmake \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -S "<path to source directory>" \
  -B "<path to build directory>"

cmake --build "<path to build directory>"

pip install <path to build directory>/py_pkg/dist/compiler_gym*.whl --force-reinstall
```

Additional optional configuration arguments:

* Enables testing.

```sh
-DCOMPILER_GYM_BUILD_TESTS=ON
```

* Builds additional tools required by some examples.

```sh
-DCOMPILER_GYM_BUILD_EXAMPLES=ON
```

* For faster rebuilds.

```sh
-DCMAKE_C_COMPILER_LAUNCHER=ccache
-DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```

* For faster linking.

```sh
-DCMAKE_EXE_LINKER_FLAGS_INIT="-fuse-ld=lld"
-DCMAKE_MODULE_LINKER_FLAGS_INIT="-fuse-ld=lld"
-DCMAKE_SHARED_LINKER_FLAGS_INIT="-fuse-ld=lld"
```

By default, CompilerGym builds LLVM from source. This takes a long time and a
lot of compute resources. To prevent this, download a pre-compiled clang+llvm
release of LLVM 10.0.0 from the [llvm-project releases
page](https://github.com/llvm/llvm-project/releases/tag/llvmorg-10.0.0), unpack
it, and pass path of the `lib/cmake/llvm` subdirectory in the archive you just
extracted to `LLVM_DIR`:

```sh
$ cmake ... \
    -DCOMPILER_GYM_LLVM_PROVIDER=external \
    -DLLVM_DIR=/path/to/llvm/lib/cmake/llvm
```

⚠️ CompilerGym requires exactly LLVM 10.0.0.

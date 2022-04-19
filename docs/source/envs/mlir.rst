MLIR Environment Reference
==========================

Under construction. In its current state as an MVP the environment exposes benchmarks
and an action space geared towards optimization of matrix multiplication.

The MLIR environment exposes configuration of MLIR optimization passes that are
related to matrix multiplication.
It compiles and runs MLIR code that computes matrix multiplication.

... code-block::

    C = A * B

where `A` is an `MxK` matrix and `B` is an `KxN` matrix.

Each episode is only 1 step long. All optimization parameters are supplied in a single action.


Environment Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~
M, N and K are benchmark parameters that initialize the environment and can not be changed.


Observation Space
~~~~~~~~~~~~~~~~~
The observation is the running time of the calculation.


Reward Space
~~~~~~~~~~~~
The reward is -running time.


Action Space
~~~~~~~~~~~~

.. code-block::

    from compiler_gym.spaces import (
        Box,
        Dict,
        Discrete,
        NamedDiscrete,
        Permutation,
        Scalar,
        SpaceSequence,
    )

    action_space = SpaceSequence(
        name="MatrixMultiplication",
        size_range=[1, 4],
        space=Dict(
            name=None,
            spaces={
                "tile_options": Dict(
                    name=None,
                    spaces={
                        "interchange_vector": Permutation(
                            name=None,
                            scalar_range=Scalar(name=None, min=0, max=2, dtype=int),
                        ),
                        "tile_sizes": Box(
                            name=None,
                            low=np.array([1] * 3, dtype=int),
                            high=np.array([2 ** 32] * 3, dtype=int),
                            dtype=np.int64,
                        ),
                        "promote": Scalar(name=None, min=False, max=True, dtype=bool),
                        "promote_full_tile": Scalar(
                            name=None, min=False, max=True, dtype=bool
                        ),
                        "loop_type": NamedDiscrete(
                            name=None,
                            items=["loops", "affine_loops"],
                        ),
                    },
                ),
                "vectorize_options": Dict(
                    name=None,
                    spaces={
                        "vectorize_to": NamedDiscrete(
                            name=None,
                            items=["dot", "matmul", "outer_product"],
                        ),
                        "vector_transfer_split": NamedDiscrete(
                            name=None,
                            items=["none", "linalg_copy", "vector_transfer"],
                        ),
                        "unroll_vector_transfers": Scalar(
                            name=None,
                            min=False,
                            max=True,
                            dtype=bool,
                        ),
                    },
                ),
            },
        ),
    )

Note that not all optimization pass configurations are valid and some will result in an error.

RL Wrapper
~~~~~~~~~~

The environment uses some non OpenAI Gym spaces in the action space. To be able
to train with off-the-shelf RL frameworks there is a wrapper (constructed using
:func:`compiler_gym.wrappers.mlir.make_mlir_rl_wrapper_env`) that converts the
action space to use only OpenAI Gym spaces.


Installation
~~~~~~~~~~~~
The environment requires LLVM 14 and it can be built only with CMake, not with Bazel.
It is incompatible with the LLVM environment due to LLVM version conflict.
To enable the MLIR environment use these CMake variables.

.. code-block::

    COMPILER_GYM_ENABLE_MLIR_ENV=ON
    COMPILER_GYM_ENABLE_LLVM_ENV=OFF

This configuration will include the MLIR environment in the `compiler_gym` Python package.
The package will be available under `${CMAKE_BINARY_DIR}/py_pkg/dist`.

The build can automatically download and build the LLVM 14 dependency.
Instead you can build against a prebuilt LLVM.
To do that pass to CMake these variables

.. code-block::

    COMPILER_GYM_LLVM_PROVIDER=external

    # path to LLVMConfig.cmake directory.
    # e.g. clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/lib/cmake/llvm
    LLVM_DIR

    # path to MLIRConfig.cmake directory.
    # e.g. clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/lib/cmake/mlir
    MLIR_DIR

    # path to ClangConfig.cmake directory
    # e.g. clang+llvm-14.0.0-x86_64-linux-gnu-ubuntu-18.04/lib/cmake/clang
    Clang_DIR

Example Usage
~~~~~~~~~~~~~

.. code-block::

    import gym
    from compiler_gym.wrappers.mlir import make_mlir_rl_wrapper_env

    env = gym.make("mlir-v0")
    wrapper = make_mlir_rl_wrapper_env(env)
    wrapper.reset()
    observation, reward, done, info = wrapper.step(wrapper.action_space.sample())

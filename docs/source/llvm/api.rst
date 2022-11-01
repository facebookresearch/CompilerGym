compiler_gym.envs.llvm
======================

The :code:`compiler_gym.envs.llvm` module contains datasets and API extensions
for the :doc:`LLVM Environments <index>`. See :class:`LlvmEnv
<compiler_gym.envs.LlvmEnv>` for the class definition.

.. contents:: Document contents:
   :local:

Constructing Benchmarks
-----------------------

.. currentmodule:: compiler_gym.envs.llvm

.. autofunction:: make_benchmark

.. autoclass:: BenchmarkFromCommandLine
   :members:

.. autoclass:: ClangInvocation
   :members:

   .. automethod:: __init__

.. autofunction:: get_system_library_flags


Datasets
--------

.. currentmodule:: compiler_gym.envs.llvm.datasets

.. autofunction:: get_llvm_datasets

.. autoclass:: AnghaBenchDataset

.. autoclass:: BlasDataset

.. autoclass:: CBenchDataset

.. autoclass:: CLgenDataset

.. autoclass:: CsmithDataset

.. autoclass:: GitHubDataset

.. autoclass:: JotaiBenchDataset

.. autoclass:: LinuxDataset

.. autoclass:: LlvmStressDataset

.. autoclass:: MibenchDataset

.. autoclass:: NPBDataset

.. autoclass:: OpenCVDataset

.. autoclass:: POJ104Dataset

.. autoclass:: TensorFlowDataset


Miscellaneous
-------------

.. currentmodule:: compiler_gym.envs.llvm

.. autofunction:: compute_observation

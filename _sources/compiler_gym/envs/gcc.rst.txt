compiler_gym.envs.gcc
======================

The :code:`compiler_gym.envs.gcc` module contains datasets and API extensions
for the :doc:`GCC Environments </envs/gcc>`. See :class:`GccEnv
<compiler_gym.envs.GccEnv>` for the class definition.

.. contents:: Document contents:
   :local:

Compiler Description
--------------------

.. currentmodule:: compiler_gym.envs.gcc.gcc

.. autoclass:: Gcc
    :special-members: __call__
    :members:

.. autoclass:: GccSpec
    :members:

.. autoclass:: Option
    :members:

.. autoclass:: GccOOption
    :members:

.. autoclass:: GccFlagOption
    :members:

.. autoclass:: GccFlagEnumOption
    :members:

.. autoclass:: GccFlagIntOption
    :members:

.. autoclass:: GccFlagAlignOption
    :members:

.. autoclass:: GccParamEnumOption
    :members:

.. autoclass:: GccParamIntOption
    :members:

Datasets
--------

.. currentmodule:: compiler_gym.envs.gcc.datasets

.. autofunction:: get_gcc_datasets

.. autoclass:: AnghaBenchDataset

.. autoclass:: CHStoneDataset

.. autoclass:: CsmithDataset

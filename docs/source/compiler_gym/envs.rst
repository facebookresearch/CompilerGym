compiler_gym.envs
=================

The :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` environment
is a drop-in replacement for the basic :code:`gym.Env` class, with
extended functionality for compilers. Some compiler services may further
extend the functionality by subclassing from
:class:`CompilerEnv <compiler_gym.envs.CompilerEnv>`. The following
environment classes are available:

.. contents::
    :local:

.. currentmodule:: compiler_gym.envs

CompilerEnv
-----------

.. autoclass:: CompilerEnv
   :members:

   .. automethod:: __init__


LlvmEnv
-------

.. autoclass:: LlvmEnv
   :members:

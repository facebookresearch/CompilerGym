compiler_gym.views
==================

The :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` exposes the
available observation spaces and reward spaces through view
objects. These views provide a flexible interface into lazily computed
values. At any point during the lifetime of an environment, any of the
available observation and reward spaces can be queried through the
:py:attr:`~compiler_gym.envs.CompilerEnv.observation` and
:py:attr:`~compiler_gym.envs.CompilerEnv.reward` attributes,
respectively.


.. currentmodule:: compiler_gym.views


ObservationView
---------------

.. autoclass:: ObservationView
   :members:

   .. automethod:: __getitem__

ObservationSpaceSpec
--------------------

.. autoclass:: ObservationSpaceSpec
   :members:
   :exclude-members: from_proto


RewardView
----------

.. autoclass:: RewardView
   :members:

   .. automethod:: __getitem__

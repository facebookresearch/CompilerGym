compiler_gym.wrappers
=====================

.. automodule:: compiler_gym.wrappers

.. contents:: Document contents:
    :local:

.. currentmodule:: compiler_gym.wrappers


Base wrappers
-------------

.. autoclass:: CompilerEnvWrapper

    .. automethod:: __init__


.. autoclass:: ActionWrapper

    .. automethod:: action

    .. automethod:: reverse_action


.. autoclass:: ObservationWrapper

    .. automethod:: observation


.. autoclass:: RewardWrapper

    .. automethod:: reward


Action space wrappers
---------------------

.. autoclass:: CommandlineWithTerminalAction

    .. automethod:: __init__


.. autoclass:: ConstrainedCommandline

    .. automethod:: __init__


.. autoclass:: TimeLimit


Datasets wrappers
-----------------

.. autoclass:: IterateOverBenchmarks

    .. automethod:: __init__


.. autoclass:: CycleOverBenchmarks

    .. automethod:: __init__


.. autoclass:: RandomOrderBenchmarks

    .. automethod:: __init__

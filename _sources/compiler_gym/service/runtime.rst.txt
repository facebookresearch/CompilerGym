compiler_gym.service.runtime
============================

.. currentmodule:: compiler_gym.service.runtime

CompilerGym uses a client/server architecture. Services provide an interface for
manipulating compiler behavior. Clients are Python frontend objects that provide
a reinforcement learning abstraction on top of the service. Communication
between the service and client is done :doc:`using RPC </rpc>`. The connection between the
client and service is managed by the :class:`CompilerGymServiceConnection
<compiler_gym.service.CompilerGymServiceConnection>` object.

.. contents:: Document contents:
    :local:


Common Runtime
--------------

.. automodule:: compiler_gym.service.runtime
    :members:


CompilerGymService
------------------

.. autoclass:: compiler_gym.service.runtime.compiler_gym_service.CompilerGymService
   :members:

    .. automethod:: __init__


BenchmarkCache
--------------

.. autoclass:: compiler_gym.service.runtime.benchmark_cache.BenchmarkCache
   :members:

    .. automethod:: __init__

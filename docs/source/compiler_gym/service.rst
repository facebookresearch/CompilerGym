compiler_gym.service
====================

.. currentmodule:: compiler_gym.service

The :code:`compiler_gym.service` module provides a client/service implementation
of the :class:`CompilerEnv <compiler_gym.envs.CompilerEnv>` interface. Services
provide an interface for manipulating compiler behavior. Clients are Python
frontend objects that provide a reinforcement learning abstraction on top of the
service. Communication between the service and client is done :doc:`using RPC
</rpc>`. The connection between the client and service is managed by the
:class:`CompilerGymServiceConnection
<compiler_gym.service.CompilerGymServiceConnection>` object.

.. contents:: Document contents:
    :local:


The CompilationSession Interface
--------------------------------

.. autoclass:: CompilationSession
   :members:

   .. automethod:: __init__


ClientServiceCompilerEnv
------------------------

.. autoclass:: compiler_gym.service.client_service_compiler_env.ClientServiceCompilerEnv
   :members:

   .. automethod:: __init__


.. autoclass:: compiler_gym.service.client_service_compiler_env.ServiceMessageConverters
   :members:

   .. automethod:: __init__


InProcessClientCompilerEnv
--------------------------

.. autoclass:: compiler_gym.service.client_service_compiler_env.InProcessClientCompilerEnv
   :members:

   .. automethod:: __init__


The connection object
---------------------

.. autoclass:: CompilerGymServiceConnection
   :members:

   .. automethod:: __init__
   .. automethod:: __call__

Configuring the connection
--------------------------

The :class:`ConnectionOpts <compiler_gym.service.ConnectionOpts>` object is used
to configure the options used for managing a service connection.

.. autoclass:: ConnectionOpts
   :members:


Exceptions
----------

In general, errors raised by the service are converted to the equivalent builtin
exception type, e.g., `ValueError` for invalid method arguments, and
`FileNotFound` for resource errors. However, some error cases are not well
covered by the builtin exception hierarchy. For those cases, we define custom
exception types, all inheriting from a base :class:`ServiceError
<compiler_gym.service.ServiceError>` class:

.. autoexception:: ServiceError

.. autoexception:: EnvironmentNotSupported

.. autoexception:: ServiceInitError

.. autoexception:: ServiceTransportError

.. autoexception:: ServiceIsClosed

.. autoexception:: SessionNotFound

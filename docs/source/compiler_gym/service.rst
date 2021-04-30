compiler_gym.service
====================

.. currentmodule:: compiler_gym.service

CompilerGym uses a client/server architecture. Services provide an interface for
manipulating compiler behavior. Clients are Python frontend objects that provide
a reinforcement learning abstraction on top of the service. Communication
between the service and client is done using RPC. The connection between the
client and service is managed by the :class:`CompilerGymServiceConnection
<compiler_gym.service.CompilerGymServiceConnection>` object.

.. contents:: Document contents:
    :local:

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

.. autoexception:: ServiceInitError

.. autoexception:: ServiceTransportError

.. autoexception:: ServiceIsClosed

.. autoexception:: SessionNotFound

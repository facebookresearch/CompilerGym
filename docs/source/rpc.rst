RPC Service Reference
=====================

This document describes the remote procedure call (RPC) client/service
architecture that CompilerGym to separate the user frontend code from the
compiler backends.

.. contents::
   :local:


How it Works
------------

The file :code:`compiler_gym/service/proto/compiler_gym_service.proto` defines a
:ref:`CompilerGymService <rpc:CompilerGymService>`, which is an interface that
can be used by compilers to expose the incremental compilation of a program as
an interactive environment. The service is defined using `gRPC
<https://grpc.io/>`_, and the individual requests and responses are defined
using `protocol buffers <https://developers.google.com/protocol-buffers>`_. The
protocol buffer schema is then used to generate bindings in a programming
language of choice. Protocol buffers support a wide range of programming
languages, allowing compiler developers to expose their optimization problems in
whatever language makes sense for them.

To use the service from C++, include the generated protocol buffer header:

.. code-block:: c++

   #include "compiler_gym/service/proto/compiler_gym_service.pb.h"

To use the service from Python, import the generated protocol buffer module:

.. code-block:: python

   import compiler_gym.service.proto


CompilerGymService
------------------

.. doxygennamespace:: CompilerGymService


Request and Reply Messages
--------------------------

.. doxygenstruct:: GetVersionRequest
   :members:

.. doxygenstruct:: GetVersionReply
   :members:

.. doxygenstruct:: GetSpacesRequest
   :members:

.. doxygenstruct:: GetSpacesReply
   :members:

.. doxygenstruct:: StartSessionRequest
   :members:

.. doxygenstruct:: StartSessionReply
   :members:

.. doxygenstruct:: ForkSessionRequest
   :members:

.. doxygenstruct:: ForkSessionReply
   :members:

.. doxygenstruct:: EndSessionRequest
   :members:

.. doxygenstruct:: EndSessionReply
   :members:

.. doxygenstruct:: StepRequest
   :members:

.. doxygenstruct:: StepReply
   :members:

.. doxygenstruct:: AddBenchmarkRequest
   :members:

.. doxygenstruct:: AddBenchmarkReply
   :members:

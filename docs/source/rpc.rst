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
language of choice.

Protocol buffers support a wide range of programming languages, allowing
compiler developers to expose their optimization problems in whatever language
makes sense for them. For Python and C++ we also provide a common runtime that
offers a higher level of abstraction and takes care of much of the boilerplate
required for RPC communication. For further details check out the :ref:`C++
<cc/compiler_gym/service:CompilationSession.h>` or :class:`Python class
<compiler_gym.service.CompilationSession>` documentation.


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

.. doxygenstruct:: SendSessionParameterRequest
   :members:

.. doxygenstruct:: SendSessionParameterReply
   :members:


Core Message Types
------------------

.. doxygenstruct:: ActionSpace
   :members:

.. doxygenstruct:: Action
   :members:

.. doxygenstruct:: ChoiceSpace
   :members:

.. doxygenstruct:: Choice
   :members:

.. doxygenstruct:: ObservationSpace
   :members:

.. doxygenstruct:: Observation
   :members:

.. doxygenstruct:: NamedDiscreteSpace
   :members:

.. doxygenstruct:: Int64List
   :members:

.. doxygenstruct:: DoubleList
   :members:

.. doxygenstruct:: ScalarRange
   :members:

.. doxygenstruct:: ScalarLimit
   :members:

.. doxygenstruct:: ScalarRangeList
   :members:

.. doxygenstruct:: SequenceSpace
   :members:

.. doxygenstruct:: Benchmark
   :members:

.. doxygenstruct:: File
   :members:

.. doxygenstruct:: BenchmarkDynamicConfig
   :members:

.. doxygenstruct:: Command
   :members:

.. doxygenstruct:: SessionParameter
    :members:

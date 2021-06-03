compiler_gym/envs/llvm/service
==============================

This directory contains the core C++ implementation of the LLVM environment for
CompilerGym. The base session is implemented by a
:code:`compiler_gym::llvm_service::LlvmSession` class, defined in
:ref:`LlvmSession.h <cc/compiler_gym/envs/llvm/service:LlvmSession.h>`.

.. contents::
   :local:

ActionSpace.h
-------------

:code:`#include "compiler_gym/envs/llvm/service/ActionSpace.h"`

.. doxygenfile:: compiler_gym/envs/llvm/service/ActionSpace.h

Benchmark.h
-----------

:code:`#include "compiler_gym/envs/llvm/service/Benchmark.h"`

.. doxygenfile:: compiler_gym/envs/llvm/service/Benchmark.h

BenchmarkFactory.h
------------------

:code:`#include "compiler_gym/envs/llvm/service/BenchmarkFactory.h"`

.. doxygenfile:: compiler_gym/envs/llvm/service/BenchmarkFactory.h

Cost.h
------

:code:`#include "compiler_gym/envs/llvm/service/Cost.h"`

.. doxygenfile:: compiler_gym/envs/llvm/service/Cost.h


LlvmSession.h
-------------

:code:`#include "compiler_gym/envs/llvm/service/LlvmSession.h"`

.. doxygenfile:: compiler_gym/envs/llvm/service/LlvmSession.h

ObservationSpaces.h
-------------------

:code:`#include "compiler_gym/envs/llvm/service/ObservationSpaces.h"`

.. doxygenfile:: compiler_gym/envs/llvm/service/ObservationSpaces.h

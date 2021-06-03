compiler_gym/service/runtime
============================

This directory contains the CompilerGym runtime that takes a
:code:`compiler_gym::CompilationSession` subclass and provides an RPC service
that can be used by the Python frontend.

.. contents::
:local:

Runtime.h
---------

:code:`#include "compiler_gym/service/runtime/Runtime.h"`

.. doxygenfile:: compiler_gym/service/runtime/Runtime.h

..
  CompilerGymService.h
  --------------------

  :code:`#include "compiler_gym/service/runtime/CompilerGymService.h"`

  .. doxygenfile:: compiler_gym/service/runtime/CompilerGymService.h

  BenchmarkCache.h
  ----------------

  :code:`#include "compiler_gym/service/runtime/BenchmarkCache.h"`

  .. doxygenfile:: compiler_gym/service/runtime/BenchmarkCache.h

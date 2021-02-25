Frequently Asked Questions
==========================

This page answers some of the commonly asked questions about CompilerGym. Have a
question not answered here? File an issue on the `GitHub issue tracker
<https://github.com/facebookresearch/CompilerGym/issues>`_.

.. contents:: Questions:
    :local:

What can I do with this?
------------------------

This projects lets you control the decisions that a compiler makes when
optimizing a program. Currently, it lets you control the selection and ordering
of optimization passes for LLVM in order to minimize the size of the code.

We wrote a small wrapper around the OpenAI gym environment which lets you step
through the optimization of a program using a text user interface. Have a play
around with it to better understand what is going on:

::

    $ pythom -m compiler_gym.bin.manual_env --env=llvm-v0


I found a bug. How do I report it?
----------------------------------

Great! Please file an issue using the `GitHub issue tracker
<https://github.com/facebookresearch/CompilerGym/issues>`_.  See
:doc:`contributing` for more details.


Do I have to use reinforcement learning?
----------------------------------------

No. We think that the the gym provides a useful abstraction for sequential
decision making. You may use any technique you wish to explore the optimization
space.


What features are going to be added in the future?
--------------------------------------------------

See :ref:`roadmap <about:roadmap>`.


How do I run this on my own program?
------------------------------------

For LLVM, you compile your program to an unoptimized LLVM bitcode file. This can
be done automatically for C/C++ programs using the :meth:`env.make_benchmark()
<compiler_gym.envs.llvm.make_benchmark>` API, or you can do this yourself using
clang:

::

    $ clang -emit-llvm -c -O0 -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes myapp.c

Then pass the path of the generated `.bc` file to the CompilerGym commandline
tools using the `--benchmark` flag, e.g.

::

    $ bazel run -c opt //compiler_gym/bin:random_search -- --env=llvm-ic-v0 \
        --benchmark=file:///$PWD/myapp.bc


I want to add a new program representation / reward signal. How do I do that?
-----------------------------------------------------------------------------

If your program representation can be computed from existing observations,
consider using the :meth:`add_derived_space()
<compiler_gym.views.ObservationSpace.add_derived_space>` API to add a derived
observation or :meth:`add_space() <compiler_gym.views.RewardView.add_space>` to
add a new reward space.

If you require modifying the underlying compiler service implementation, fork
this project and build it from source (see :doc:`installation`). Then modify the
C++ service implementation for the compiler that you are interested in. The
service codebase is located at :code:`compiler_gym/envs/$COMPILER/service`,
where :code:`$COMPILER` is the name of the compiler service you would wish to
modify, e.g. llvm. Once done, send us a pull request!

I don't think compiler optimization is a sequential decision process. How is that enabled?
------------------------------------------------------------------------------------------

Compilers package individual optimizations as "passes" which are then applied
in a sequential order. Usually the order inside a compiler is fixed (e.g.
`take a look at how LLVM does it <https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/IPO/PassManagerBuilder.cpp#L517-L922>`_).
CompilerGym replaces that fixed order with a sequential decision process where
any compiler loop is allowed to be applied at any stage.

I wonder when does CompilerGym consider an episode “done”?
----------------------------------------------------------

The compiler itself doesn't have a signal for termination. Optimizations are
like rewrite rules, it is up to the user to decide when no more improvement
can be achieved from further rewrites. E.g. for simple random search we can
use "patience" `[1] <https://github.com/facebookresearch/CompilerGym/blob/development/compiler_gym/bin/random_search.py#L33-L40/>`_.
The only exception is if the compiler crashes, or the code ends up in an
unexpected state - we have to abort. This happens.

Should I always try different actions, or repeating the same action multiple times can improve the results?
----------------------------------------------------------------------------------------------------------

Some actions such as dead code elminiation (-dce), are typically called multiple
times after other optimization passes. So yes, repeating the same action in
different context can bring improvements.

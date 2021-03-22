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

Once you get the hang of things, try submitting your best algorithm to our
`leaderboards <https://github.com/facebookresearch/CompilerGym#leaderboards>`_!


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


Is compiler optimization really a sequential decision process?
--------------------------------------------------------------

Compilers frequently package individual transformations as "optimization passes"
which are then applied in a sequential order. Usually this order is fixed (e.g.
`realworld example
<https://github.com/llvm/llvm-project/blob/71a8e4e7d6b947c8b954ec0763ff7969b3879d7b/llvm/lib/Transforms/IPO/PassManagerBuilder.cpp#L517-L922>`_).
CompilerGym replaces that fixed order with a sequential decision process where
any pass may be applied at any stage.


When does a compiler enviornment consider an episode “done”?
------------------------------------------------------------

The compiler itself doesn't have a signal for termination. Actions are like
rewrite rules, it is up to the user to decide when no more improvement can be
achieved from further rewrites. E.g. for simple random search we can use
`"patience"
<https://github.com/facebookresearch/CompilerGym/blob/8fa65c232d2bf6a7347af44565579c60775162ac/compiler_gym/bin/random_search.py#L33-L40>`_.
The only exception is if the compiler crashes, or the code ends up in an
unexpected state - we have to abort. This happens.


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


Should I always try different actions?
--------------------------------------

Some optimization actions may be called multiple times after other actions. An
example of this is `dead code elimination
<https://en.wikipedia.org/wiki/Dead_code_elimination>`_, which can be used to
"clean up mess" generated from a previous action. So repeating the same action
in different context can bring improvements.


I updated with "git pull" and now it doesn't work
-------------------------------------------------

The first thing to is to re-run :code:`make init` to ensure that you have the
correct development depencies installed, as those can change between releases.

If that doesn't fix the problem, feel free to
`file an issue <https://github.com/facebookresearch/CompilerGym/issues>`_, but
note that the
`development <https://github.com/facebookresearch/CompilerGym/commits/development>`_
branch is the bleeding edge and may contain features that have not yet reached
stability. If you would like to build from source but do not require the
latest feature set, use the
`stable <https://github.com/facebookresearch/CompilerGym/commits/stable>`_
branch which lags to the latest release with hotfixes.

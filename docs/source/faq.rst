Frequently Asked Questions
==========================

This page answers some of the commonly asked questions about
CompilerGym. Have a question not answered here? File an issue on the
`GitHub issue tracker <https://github.com/facebookresearch/CompilerGym/issues>`_.

.. contents:: Questions:
    :local:

What can I do with this?
------------------------

This projects lets you control the decisions that a compiler makes
when optimizing a program. Currently, it lets you control the
selection and ordering of optimization passes for LLVM in order to
minimize the size of the code.

We wrote a small wrapper around the OpenAI gym environment which lets
you step through the optimization of a program using a text user
interface. Have a play around with it to better understand what is
going on:

::

    $ pythom -m compiler_gym.bin.manual_env --env=llvm-v0


I found a bug. How do I report it?
----------------------------------

Great! Please file an issue using the `GitHub issue tracker
<https://github.com/facebookresearch/CompilerGym/issues>`_.  See
:doc:`contributing` for more details.


Do I have to use reinforcement learning?
----------------------------------------

No. We think that the the gym provides a useful abstraction for
sequential decision making. You may use any technique you wish to
explore the optimization space.


What features are going to be added in the future?
--------------------------------------------------

See :ref:`roadmap <about:roadmap>`.


How do I run this on my own program?
------------------------------------

For LLVM, you compile your program to an unoptimized LLVM bitcode
file. For C/C++ programs, you can do this using clang by running:

::

    $ clang -emit-llvm -c -O0 -Xclang -disable-O0-optnone -Xclang -disable-llvm-passes myapp.c

Then pass the path of the generated `.bc` file to the CompilerGym
commandline tools using the `--program_data` flag, e.g.

::

    $ bazel run -c opt //compiler_gym/bin:brute_force -- --env=llvm-v0 \
        --program_data=file:///$PWD/myapp.bc


I want to add a new program representation / reward signal. How do I do that?
-----------------------------------------------------------------------------

Fork this project and build it from source (see
:doc:`installation`). Then modify the C++ service implementation for
the compiler that you are interested in. The service codebase is
located at :code:`compiler_gym/envs/$COMPILER/service`, where
:code:`$COMPILER` is the name of the compiler service you would wish
to modify, e.g. llvm. Once done, send us a pull request!

Frequently Asked Questions
==========================

This page answers some of the commonly asked questions about CompilerGym. Have a
question not answered here? File an issue on the `GitHub issue tracker
<https://github.com/facebookresearch/CompilerGym/issues>`_.

.. contents:: Topics:
    :local:

What can I do with this?
------------------------

CompilerGym lets you control the decisions that a compiler makes when optimizing
a program. Currently, it lets you control the selection and ordering of
optimization passes for LLVM, the command line flags for the GCC compiler, and
the order of loop nests for CUDA. The goal is to steer the compiler towards
producing the best compiled program, as determined by a reward signal that
measures some property of the program such as its code size or execution time.


Do I have to use reinforcement learning?
----------------------------------------

No. We think that the the gym provides a useful abstraction for sequential
decision making. You may use any technique you wish to explore the optimization
space. Researchers have had success using search techniques, genetic algorithms,
supervised and unsupervised machine learning, and deep reinforcement learning.
Take a look at `this paper <https://chriscummins.cc/pub/2020-fdl.pdf>`_ for a
brief introduction to the field.


What features are going to be added in the future?
--------------------------------------------------

See :ref:`roadmap <about:roadmap>`.


Development
-----------


I found a bug. How do I report it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Great! Please file an issue using the `GitHub issue tracker
<https://github.com/facebookresearch/CompilerGym/issues>`_.  See
:doc:`contributing` for more details.


I updated with "git pull" and now it doesn't work
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first thing to is to re-run :code:`make init` to ensure that you have the
correct development dependencies installed, as those can change between
releases. Then run :code:`make distclean` to tidy up any build artifacts from
the old version.

If that doesn't fix the problem, feel free to
`file an issue <https://github.com/facebookresearch/CompilerGym/issues>`_, but
note that the
`development <https://github.com/facebookresearch/CompilerGym/commits/development>`_
branch is the bleeding edge and may contain features that have not yet reached
stability. If you would like to build from source but do not require the
latest feature set, use the
`stable <https://github.com/facebookresearch/CompilerGym/commits/stable>`_
branch which lags to the latest release with hotfixes.


I want to add a new program representation / reward signal. How do I do that?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your reward or observation is a transformation of an existing space, consider
using the :mod:`compiler_gym.wrappers` module to define a wrapper that performs
the translation from the base space.

If your reward or observation requires combining multiple existing spaces,
consider using :meth:`add_derived_space()
<compiler_gym.views.ObservationView.add_derived_space>` or :meth:`add_space()
<compiler_gym.views.RewardView.add_space>`.

If you require modifying the underlying compiler service implementation, fork
this project and build it from source (see `installation
<https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md>`_).
Then modify the service implementation for the compiler that you are interested
in. The service codebase is located at
:code:`compiler_gym/envs/$COMPILER/service`, where :code:`$COMPILER` is the name
of the compiler service you would wish to modify, e.g. llvm. Once done, send us
a pull request!

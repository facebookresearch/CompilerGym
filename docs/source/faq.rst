Frequently Asked Questions
==========================

This page answers some of the commonly asked questions about CompilerGym. Have a
question not answered here? File an issue on the `GitHub issue tracker
<https://github.com/facebookresearch/CompilerGym/issues>`_.

.. contents:: Topics:
    :local:


Usage
-----

What can I do with this?
~~~~~~~~~~~~~~~~~~~~~~~~

CompilerGym lets you control the decisions that a compiler makes when optimizing
a program. Currently, it lets you control the selection and ordering of
optimization passes for LLVM, the command line flags for the GCC compiler, and
the order of loop nests for CUDA. The goal is to steer the compiler towards
producing the best compiled program, as determined by a reward signal that
measures some property of the program such as its code size or execution time.


Do I have to use reinforcement learning?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

No. We think that the the gym provides a useful abstraction for sequential
decision making. You may use any technique you wish to explore the optimization
space. Researchers have had success using search techniques, genetic algorithms,
supervised and unsupervised machine learning, and deep reinforcement learning.
Take a look at `this paper <https://chriscummins.cc/pub/2020-fdl.pdf>`_ for a
brief introduction to the field.



Why does my environment's step() function return "done"?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The third element of the tuple returned by :meth:`env.step()
<compiler_gym.envs.CompilerEnv.step>` is a boolean that indicates whether the
current episode is "done". There are two reasons why an episode may be "done":

1. A terminal state has been reached, as defined by the dynamics of the
   environment. This could be because there are no further decisions to be made,
   or because of an artificial limit to the episode length such as provided by
   the :class:`TimeLimit <compiler_gym.wrappers.TimeLimit>` and
   :class:`CommandlineWithTerminalAction
   <compiler_gym.wrappers.CommandlineWithTerminalAction>` wrappers.

2. The environment has encountered an unrecoverable error and can no longer
   proceed with the episode. This could be because of an error such as a
   compiler crashing, a timeout from an action that takes too long, or an
   overloaded system causing an out-of-memory error.

In case of an unrecoverable error, CompilerGym will provide a description of the
error through a string :code:`error_details` in the returned :code:`info` dict:

.. code-block:: python

    >>> import compiler_gym
    >>> env = compiler_gym.make("llvm-v0")
    >>> env.reset()
    >>> for _ in range(1000):
    ...     observation, reward, done, info = env.step(env.action_space.sample())
    ...     if done:
    ...         print(info.get("error_details"))
    ...         env.reset()

In either case, calling :meth:`env.reset()
<compiler_gym.envs.CompilerEnv.reset>` will reset the environment and start a
new episode.


Where does CompilerGym store files?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

CompilerGym is a python library, and is located in the `site packages
<https://docs.python.org/3/library/site.html#site.getsitepackages>`_ directory.
Remove it using :code:`pip uninstall compiler_gym`. In addition, CompilerGym
uses the following file system locations to store files:

- :code:`~/.local/cache/compiler_gym` is used to cache files such as downloaded
  datasets. Set environment variable :code:`$COMPILER_GYM_CACHE` to override
  this default location.

- :code:`~/.local/share/compiler_gym` is used to store additional datasets and
  files that are not included in the core CompilerGym library. Set environment
  variable :code:`$COMPILER_GYM_SITE_DATA` to override this default location.

- :code:`/dev/shm/compiler_gym_${USER}` is used as an in-memory cache on Linux
  systems which support it. Files in this cache are should not outlive the
  lifespan of the CompilerGym environments that created them. Set environment
  variable :code:`$COMPILER_GYM_TRANSIENT_CACHE` to override this default
  location.

- :code:`~/logs/compiler_gym` is used by some of the example scripts to store
  logs and experiment artifacts. Set environment variable
  :code:`$COMPILER_GYM_LOGS` to override this default location.

It is perfectly safe to delete all of the above directories, so long as there
are no active Python processes using CompilerGym. After deleting the above
directories, you may notice a delay the next time you launch a CompilerGym
environment as files and datasets are re-downloaded and unpacked.


How do I debug crashes or errors?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The first step is to produce a minimal, reproducible example. The easiest way to
do this is usually to copy your code into a new file and to iteratively remove
as many dependencies and chunks of code as possible while still preserving the
error behavior of interest. Second, you can inspect CompilerGym's logging
output.

CompilerGym uses Python's built in logging mechanism, but emits warning and
error messages sparingly. In normal use, the logging messages from CompilerGym
will not be seen by users. To enable these messages to be logged to standard out
insert the following snippet at the start of your script, before instantiating
any CompilerGym environments:

.. code-block:: python

    import logging
    logging.basicConfig(level=logging.DEBUG)

    # ... rest of script

.. note::

    CompilerGym uses per-module loggers. For fine-grained control over logging,
    access the :code:`compiler_gym` logger or its children.

Additionally, even-more-verbose logging can be enabled by setting the environment
variable :code:`COMPILER_GYM_DEBUG` to 4:

.. code-block:: python

    import logging
    import os
    os.environ["COMPILER_GYM_DEBUG"] = "4"
    logging.basicConfig(level=logging.DEBUG)

    # ... rest of script

Inspecting the verbose logs may help understand what is going on, and is
incredibly helpful when reporting bugs. See :ref:`report-bug`.


.. _report-bug:

I found a bug. How do I report it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Great! Please file an issue using the `GitHub issue tracker
<https://github.com/facebookresearch/CompilerGym/issues>`_.  See
:doc:`contributing` for more details.


Development
-----------


What features are going to be added in the future?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

See :ref:`roadmap <about:roadmap>`.


I want to modify one of the environments, where do I start?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many modifications to CompilerGym environments can be achieved through
:mod:`wrappers <compiler_gym.wrappers>`. For example, you can use the existing
wrappers to artificially limit the length of episodes using :class:`TimeLimit
<compiler_gym.wrappers.TimeLimit>`, constrain the actions available to the agent
through :class:`ConstrainedCommandline
<compiler_gym.wrappers.ConstrainedCommandline>`, or randomize the benchmark that
is selected on :code:`reset()` using :class:`RandomOrderBenchmarks
<compiler_gym.wrappers.RandomOrderBenchmarks>`.

Many new types of modular transformations can be implemented by extending the
base wrapper classes. For example, custom reward shaping can be implemented by
implementing the :code:`reward()` method in :class:`RewardWrapper
<compiler_gym.wrappers.RewardWrapper>`.

For more invasive changes, you may need to modify the underlying environment
implementation. To do that, fork this project and build it from source (see
`installation
<https://github.com/facebookresearch/CompilerGym/blob/development/INSTALL.md>`_).
Then modify the compiler service implementation for the compiler that you are
interested in. The service codebase is located at
:code:`compiler_gym/envs/$COMPILER/service`, where :code:`$COMPILER` is the name
of the compiler service you would wish to modify, e.g.
:code:`compiler_gym/envs/llvm/service` for the :doc:`LLVM environment
<llvm/index>`. Once done, send us a pull request!


I want to add a new compiler environment, where do I start?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add a new environment, you must implement the :class:`CompilationSession
<compiler_gym.service.CompilationSession>` interface to provide a new
compilation service, and then register this service with the CompilerGym
frontend. The whole process takes less than 200 lines of code. Full end-to-end
examples are provided for both Python and C++ in the `examples directory
<https://github.com/facebookresearch/CompilerGym/tree/development/examples/example_compiler_gym_service>`_. Once done, send us a pull request!


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

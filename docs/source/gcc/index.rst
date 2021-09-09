GCC Environment Reference
==========================

`GCC <https://gcc.gnu.org/>`_ is a production-grade compiler for C and C++ used
throughout industry.

CompilerGym exposes the GCC optimizations for reinforcement learning through an
:class:`GccEnv <compiler_gym.envs.gcc.GccEnv>` environment.

The environment works with any version of GCC from 5 up to and including the
current version at time of writing, 11.2. The environment uses Docker images to
enable hassle free install and consistency across machines. Alternatively, any
local installation of the compiler can be used. This selection is made by simple
string specifier of the path or docker image name.

The GCC binary to use can be specified by adding a `gcc_bin` argument when
creating the environment.  This can either be a path or executable name, or if
using a docker image, then the name of the image, preceded by `docker:`. The
default is `docker:gcc:11.2.0`. Docker images are automatically downloaded if
possible.

For example, the following creates a GCC environment using GCC 5 in a docker
image:

    >>> env = compiler_gym.make("gcc-v0", gcc_bin="docker:gcc:5")

While this creates one using an existing `gcc-11` command on the system:

    >>> env = compiler_gym.make("gcc-v0", gcc_bin="gcc-11")

(Note that on Macs, the default `gcc` command is not actually GCC but points to
LLVM. Using `gcc_bin="gcc"` on a Mac is unlikely to work.)


.. contents:: Overview:
    :local:


Datasets
--------

We provide several datasets of open-source LLVM-IR benchmarks for use:

+----------------------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
| Dataset                    | Num. Benchmarks [#f1]_   | Description                                                                                                                                                                                                        | Validatable [#f2]_   |
+============================+==========================+====================================================================================================================================================================================================================+======================+
| benchmark://anghabench-v1  | 1,041,333                | Compile-only C/C++ functions extracted from GitHub [`Homepage <http://cuda.dcc.ufmg.br/angha/>`__, `Paper <https://homepages.dcc.ufmg.br/~fernando/publications/papers/FaustinoCGO21.pdf>`__]                      | No                   |
+----------------------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
| benchmark://chstone-v0     | 12                       | Benchmarks for C-based High-Level Synthesis [`Homepage <http://www.ertl.jp/chstone/>`__, `Paper <http://www.yxi.com/applications/iscas2008-300_1027.pdf>`__]                                                       | No                   |
+----------------------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
| generator://csmith-v0      | ∞                        | Random conformant C99 programs [`Homepage <https://embed.cs.utah.edu/csmith/>`__, `Paper <http://web.cse.ohio-state.edu/~rountev.1/5343/pdf/pldi11.pdf>`__]                                                        | No                   |
+----------------------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+
| Total                      | 1,041,345                |                                                                                                                                                                                                                    |                      |
+----------------------------+--------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------+

.. [#f1] Values are for the Linux datasets. Some of the datasets contain fewer
         benchmarks on macOS.
.. [#f2] A **validatable** dataset is one where the behavior of the benchmarks
         can be checked by compiling the programs to binaries and executing
         them. If the benchmarks crash, or are found to have different behavior,
         then validation fails. This type of validation is used to check that
         the compiler has not broken the semantics of the program.
         See :mod:`compiler_gym.bin.validate`.

All of the above datasets are available for use with the GCC environment. See
:ref:`compiler_gym.envs.gcc.datasets <gcc/api:Datasets>` for API details.

Observation Spaces
------------------

We provide several observation spaces for GCC.

Each observation is accessible from the environment's `observation` field:

    >>> env.observation["asm_size"]
    36102

Each of these observations is also directly accessible from the environment:

    >>> env.asm_size
    36102


Source, RTL, Assembly, Object Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------+---------+-----------------------------------------------------------+
| Observation space        | Shape   | Description                                               |
+==========================+=========+===========================================================+
| source                   | `str`   | Preprocessed C or C++ source code                         |
+--------------------------+---------+-----------------------------------------------------------+
| rtl                      | `str`   | Register  TransferLanguage code at the end of compilation |
+--------------------------+---------+-----------------------------------------------------------+
| asm                      | `str`   | Assembly code                                             |
+--------------------------+---------+-----------------------------------------------------------+
| obj                      | `bytes` | Binary of the object file                                 |
+--------------------------+---------+-----------------------------------------------------------+

These four spaces return the appropriate string or bytes representation of the
program state. These serialized representations can be accessed through the
`observation` field:

    >>> env.observation["asm"]
    '; ModuleID = \'\t.file\t"src.c"\n\t...'

Assembly and Object Code Sizes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------+---------+-----------------------------------------------------------+
| Observation space        | Shape   | Description                                               |
+==========================+=========+===========================================================+
| asm_size                 | `int`   | Number of bytes in the assembly code                      |
+--------------------------+---------+-----------------------------------------------------------+
| obj_size                 | `int`   | Number of bytes in the object code                        |
+--------------------------+---------+-----------------------------------------------------------+

Gets the number of bytes in the assembly and object codes. If only interested in
these sizes, then less information is transferred between server and client this
way.

Example values:

    >>> env.observation["asm_size"]

Assembly and Object Code Hashes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

+--------------------------+---------+-----------------------------------------------------------+
| Observation space        | Shape   | Description                                               |
+==========================+=========+===========================================================+
| asm_hash                 | `str`   | MD5 hash of the assembly code                             |
+--------------------------+---------+-----------------------------------------------------------+
| obj_hash                 | `str`   | MD5 hash of the object code                               |
+--------------------------+---------+-----------------------------------------------------------+

Gets the MD5 hash the assembly and object codes.

Example values:

    >>> env.observation["asm_hash"]
    'f4921de395b026a55eab3844c8fe43dd'

Instruction Counts
~~~~~~~~~~~~~~~~~~

+--------------------------+----------------------------------------------------------------------------------------+
| Observation space        | Shape            | Description                                                         |
+==========================+==================+=====================================================================+
| instruction_counts       | `Dict[str, int]` | A map of instruction name to count as appears in the assembly file. |
+--------------------------+------------------+---------------------------------------------------------------------+

This observation first assembles the code. Then it counts the number of each
instruction type in the assembly, including pseudo-instructions and returns that
as dictionary. If there are no instructions of a given type, then there will be
no entry for that instruction type.

Example values:

    >>> env.observation["instruction_counts"]
    {'.file': 1, '.text': 4, '.globl': 110, '.bss': 8, '.align': 95,
     '.type': 110, '.size': 110, '.zero': 83, '.section': 10, '.long': 502,
     '.cfi': 91, 'pushq': 16, 'movq': 150, 'movl': 575, 'cmpl': 30, 'js': 7,
     'jmp': 24, 'negl': 5, 'popq': 11, 'ret': 15, 'subq': 15, 'leaq': 40,
     'movslq': 31, 'cltq': 67, 'imulq': 27, 'addq': 17, 'addl': 44, 'jle': 21,
     'sarq': 20, 'call': 34, 'subl': 7, 'sarl': 9, 'testl': 1, 'cmovns': 2,
     'jge': 3, 'sall': 2, 'orl': 1, 'leave': 4, 'andl': 2, 'nop': 7, 'cmpq': 1,
     'salq': 7, 'jns': 2, 'jne': 1, 'testq': 4, 'negq': 1, 'shrl': 2,
     '.string': 1, 'je': 2, '.ident': 1}


Choices
~~~~~~~

+--------------------------+-------------+------------------------------------------------+
| Observation space        | Shape       | Description                                    |
+==========================+=============+================================================+
| choices                  | `List[int]` | The current state of all optimization settings |
+--------------------------+-------------+------------------------------------------------+

This observation gives a list of all the choices that are currently made for the
optimization settings.

The number of optimization settings varies depending on which version of GCC is
being used. The space of options can be found from the `gcc_spec` field of an
environment:

    >>> env.gcc_spec.options
    [<GccOOption values=[0,1,2,3,fast,g,s]>, <GccFlagOption name=aggressive-loop-optimizations>, ... ]

Each option has some number of possible values. For example, the `-O` setting
which gives coarse groupings of optimizations can take any of the seven forms:
`-O0`, `-O1`, `-O2`, `-O3`, `-Ofast`, `-Og`, `-Os`. Additionally, a setting
might be missing from the command line. As another example, the second option
in the list above can be one of `-faggressive-loop-optimizations`,
`-fno-aggressive-loop-optimizations`, or missing.

Each option, then, can take a value from `[-1, cardinality]`, where `-1`
indicates that it is missing, and any other number indicates that choices from
the option.

So, if the choices are `[4, 0, -1, -1, ...]` (i.e. all but the first two are
`-1`), then this will correspond to command line arguments of:

    -Ofast -faggressive-loop-optimizations

Example values:

    >>> env.observation["choices"]
    [4, 0, -1, -1, ...]

This observation can be read directly via a property, like the other
observations. That property can also be set which will change the choices of the
current optimization settings.

    >>> env.choices = [-1] * len(env.gcc_spec.options)
    >>> env.choices
    [-1, -1, -1, -1, ...]

Command Line
~~~~~~~~~~~~

+--------------------------+-------------+------------------------------------------------+
| Observation space        | Shape       | Description                                    |
+==========================+=============+================================================+
| command_line             | `str`       | Command line synopsis of the current state.    |
+--------------------------+-------------+------------------------------------------------+

This observation gives the command line corresponding to the current choices.

Note that the command line will include a command string for the particular
version of GCC that is being used as well as input and output files

Example:

    >>> env.observation["command_line"]
    'docker:gcc:11.2.0 -w -c src.c -o obj.o'


.. _reward:

Reward Spaces
-------------

The reward spaces for the GCC environment in the CompilerGym are simple
wrappers over two of the observations, namely `asm_size` and `obj_size`. The
reward is the change in that value since the last action.

+------------------------+-------------+------------------+
| Reward space           | Range       | Deterministic?   |
+========================+=============+==================+
| asm_size               | (-inf, inf) | Yes              |
+------------------------+-------------+------------------+
| obj_size               | (-inf, inf) | Yes              |
+------------------------+-------------+------------------+

Action Space
------------

While the LLVM phase ordering action space is unbounded as passes may be
executed forever, the number of GCC command line configurations is bounded.
GCC’s action space consists of all the available optimization flags and
parameters that can be specified from the command line. These are automatically
extracted from the “help” documentation of whichever GCC version is used. For
GCC 11.2.0, the optimization space includes 502 options:

- the six `-O<n>` flags, e.g. `-O0`, `-O3`, `-Ofast`, `-Os`.
- 242 flags such as `-fpeel-loops`, each of which may be missing, present, or
  negated (e.g. `-fno-peel-loops`). Some of these flags may take integer or
  enumerated arguments which are also included in the space.
- 260 parameterized command line flags such as
  `--param inline-heuristics-hint-percent=<number>`. The number of options for each of
  these varies. Most take numbers, a few take enumerated values. The GCC action
  space is determined automatically from whichever version of GCC is being used.

This gives a finite optimization space with a modest size
of approximately 10^4461. Earlier versions of GCC report their parameter spaces
less clearly and so the tool finds smaller spaces when pointed at those. For
example, on GCC 5, the optimization space is only 10^430.


The first action space is intended to make it easy for RL tools that operate on
a flat list of categorical actions. For every option with a cardinality of
fewer than ten, we provide actions that directly set the choice for that
action. For options with greater cardinalities we provide actions that add and
subtract 1, 10, 100, and 1000to the choice integer corresponding to the option.
For GCC 11.2.0, this creates a set of 2281 actions that can modify the choices
of the current state.

So, for example, in GCC 11.2.0, the first option is the `-O` option. This has 7
possible settings, other than missing: `-O0`, `-O1`, `-O2`, `-O3`, `-Ofast`,
`-Og`, `-Os`. Since this is fewer than ten, there is a corresponding action for
each. Similarly, there are action for each of the normal GCC flags, like
`-fpeel-loops` and `-fno-peel-loops`. Parameters often have more than ten
options, so there will be actions to bump values up and down, for example, ``

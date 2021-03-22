LLVM Environment Reference
==========================

`LLVM <https://llvm.org/>`_ is a production-grade compiler used throughout
industry. It defines a machine independent intermediate representation (IR), and
comprises a family of tools with frontends for C, C++, OpenCL, and many other
languages.

CompilerGym exposes the LLVM IR optimizer for reinforcement learning through an
:class:`LlvmEnv <compiler_gym.envs.LlvmEnv>` environment.

.. contents:: Overview:
    :local:


Datasets
--------

We provide several datasets of open-source LLVM-IR benchmarks for download:

+------------------------+--------------+-----------------+---------------------+-------------------+
| Dataset                | License      | Num. Benchmarks | Validatable? [#f1]_ | Difficulty [#f2]_ |
+========================+==============+=================+=====================+===================+
| blas-v0                | BSD 3-Clause | 300             | No                  | 0.3               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| cBench-v1              | BSD 3-Clause | 23              | Partial             | 0.8               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| github-v0              | CC BY 4.0    | 50,708          | No                  | 0.7               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| linux-v0               | GPL-2.0      | 13,920          | No                  | 0.4               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| mibench-v0             | BSD 3-Clause | 40              | No                  | 0.8               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| npb-v0                 | NASA v1.3    | 122             | No                  | 0.4               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| opencv-v0              | Apache 2.0   | 442             | No                  | 0.3               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| poj104-v0              | BSD 3-Clause | 49,628          | No                  | 0.7               |
+------------------------+--------------+-----------------+---------------------+-------------------+
| tensorflow-v0          | Apache 2.0   | 1,985           | No                  | 0.3               |
+------------------------+--------------+-----------------+---------------------+-------------------+

.. [#f1] A **validatable** dataset is one where the behavior of the benchmarks
         can be checked by compiling the programs to binaries and executing
         them. If the benchmarks crash, or are found to have different behavior,
         then validation fails. This type of validation is used to check that
         the compiler has not broken the semantics of the program.
         See :mod:`compiler_gym.bin.validate`.
.. [#f2] The **difficulty** of a dataset is an indicator of how likely a random
         policy is to outperform the default compiler policy in a fixed amount
         of time. A lower difficulty shows that a random policy is more likely
         to succeed. It is a crude characterization metric that does not take
         into account factors such as the diversity of programs, the complexity
         of the optimization space, etc. The difficulty values in this table
         were estimated using 2000 random trials and a fixed time budget of 30
         seconds.

Install these datasets using the :mod:`compiler_gym.bin.datasets` command line
tool, or programatically using
:meth:`CompilerEnv.require_datasets() <compiler_gym.envs.CompilerEnv.require_datasets>`:

    >>> env = gym.make("llvm-v0")
    >>> env.require_datasets(["tensorflow-v0", "npb-v0"])


Observation Spaces
------------------

We provide several observation spaces for LLVM based on published compiler
research.


LLVM-IR
~~~~~~~

+--------------------------+-------------------------+
| Observation space        | Shape                   |
+==========================+=========================+
| Ir                       | `str_list<>[0,inf])`    |
+--------------------------+-------------------------+
| BitcodeFile              | `str_list<>[0,4096.0])` |
+--------------------------+-------------------------+

A serialized representation of the LLVM-IR can be accessed as a string through
the :code:`Ir` observation space:

    >>> env.observation["Ir"]
    '; ModuleID = \'benchmark://npb-v0/50\'\n ..."use-soft-float"="false" }\n'

Alternatively the module can be serialized to a bitcode file on disk:

    >>> env.observation["BitcodeFile"]
    '/home/user/.cache/compiler_gym/service/2020-12-21T11:55:41.716711-6f4f0669/module-5a8b9fcf.bc'

.. note::
    Files generated by the :code:`BitcodeFile` observation space are put in a
    temporary directory that is removed when :meth:`env.close() <compiler_gym.envs.CompilerEnv.close>` is called.


Autophase
~~~~~~~~~

+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Observation space        | Shape                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
+==========================+=================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================================+
| Autophase                | `Box(0, 9223372036854775807, (56,), int64)`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| AutophaseDict            | `Dict(ArgsPhi:int<0,inf>, BB03Phi:int<0,inf>, BBHiPhi:int<0,inf>, BBNoPhi:int<0,inf>, BBNumArgsHi:int<0,inf>, BBNumArgsLo:int<0,inf>, BeginPhi:int<0,inf>, BlockLow:int<0,inf>, BlockMid:int<0,inf>, BranchCount:int<0,inf>, CriticalCount:int<0,inf>, NumAShrInst:int<0,inf>, NumAddInst:int<0,inf>, NumAllocaInst:int<0,inf>, NumAndInst:int<0,inf>, NumBitCastInst:int<0,inf>, NumBrInst:int<0,inf>, NumCallInst:int<0,inf>, NumEdges:int<0,inf>, NumGetElementPtrInst:int<0,inf>, NumICmpInst:int<0,inf>, NumLShrInst:int<0,inf>, NumLoadInst:int<0,inf>, NumMulInst:int<0,inf>, NumOrInst:int<0,inf>, NumPHIInst:int<0,inf>, NumRetInst:int<0,inf>, NumSExtInst:int<0,inf>, NumSelectInst:int<0,inf>, NumShlInst:int<0,inf>, NumStoreInst:int<0,inf>, NumSubInst:int<0,inf>, NumTruncInst:int<0,inf>, NumXorInst:int<0,inf>, NumZExtInst:int<0,inf>, TotalBlocks:int<0,inf>, TotalFuncs:int<0,inf>, TotalInsts:int<0,inf>, TotalMemInst:int<0,inf>, UncondBranches:int<0,inf>, binaryConstArg:int<0,inf>, const32Bit:int<0,inf>, const64Bit:int<0,inf>, morePreds:int<0,inf>, numConstOnes:int<0,inf>, numConstZeroes:int<0,inf>, onePred:int<0,inf>, onePredOneSuc:int<0,inf>, onePredTwoSuc:int<0,inf>, oneSuccessor:int<0,inf>, returnInt:int<0,inf>, testUnary:int<0,inf>, twoEach:int<0,inf>, twoPred:int<0,inf>, twoPredOneSuc:int<0,inf>, twoSuccessor:int<0,inf>)` |
+--------------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

The Autophase observation space is a 56-dimension integer feature vector
summarizing the static LLVM-IR representation. It is described in:

    Haj-Ali, A., Huang, Q. J., Xiang, J., Moses, W., Asanovic, K., Wawrzynek,
    J., & Stoica, I. (2020).
    `AutoPhase: Juggling HLS phase orderings in random forests with deep reinforcement learning <https://proceedings.mlsys.org/paper/2020/file/4e732ced3463d06de0ca9a15b6153677-Paper.pdf>`_.
    Proceedings of Machine Learning and Systems, 2, 70-81.

Use the :code:`Autophase` observation space to access the feature vectors as an
`np.array <https://numpy.org/doc/stable/reference/generated/numpy.array.html>`_,
and :code:`AutophaseDict` to receive them as a self-documented dictionary, keyed
by the name of each feature.

Example values:


    >>> env.observation["Autophase"]
    array([   0,    0,   26,   25,    1,   26,   10,    1,    8,   10,    0,
              0,    0,   37,    0,   36,    0,    2,   46,  175, 1664, 1212,
            263,   26,  193,    0,   59,    6,    0,    3,   32,    0,   36,
             10, 1058,   10,    0,  840,    0,    0,    0,    1,  416,    0,
              0,  148,   60,    0,    0,    0,   37, 3008, 2062,    9,    0,
           1262])
    >>> env.observation["AutophaseDict"]
    {'BBNumArgsHi': 0, 'BBNumArgsLo': 0, 'onePred': 26, 'onePredOneSuc': 25,
     'onePredTwoSuc': 1, 'oneSuccessor': 26, 'twoPred': 10, 'twoPredOneSuc': 1,
     'twoEach': 8, 'twoSuccessor': 10, 'morePreds': 0, 'BB03Phi': 0,
     'BBHiPhi': 0, 'BBNoPhi': 37, 'BeginPhi': 0, 'BranchCount': 36,
     'returnInt': 0, 'CriticalCount': 2, 'NumEdges': 46, 'const32Bit': 175,
     'const64Bit': 1664, 'numConstZeroes': 1212, 'numConstOnes': 263,
     'UncondBranches': 26, 'binaryConstArg': 193, 'NumAShrInst': 0,
     'NumAddInst': 59, 'NumAllocaInst': 6, 'NumAndInst': 0, 'BlockMid': 3,
     'BlockLow': 32, 'NumBitCastInst': 0, 'NumBrInst': 36, 'NumCallInst': 10, ... }

Inst2vec
~~~~~~~~

+--------------------------+--------------------------+
| Observation space        | Shape                    |
+==========================+==========================+
| Inst2vec                 | `ndarray_list<>[0,inf])` |
+--------------------------+--------------------------+
| Inst2vecEmbeddingIndices | `int32_list<>[0,inf])`   |
+--------------------------+--------------------------+
| Inst2vecPreprocessedText | `str_list<>[0,inf])`     |
+--------------------------+--------------------------+

The inst2vec observation space represents LLVM-IR as sequence of embedding
vectors, one per LLVM statement, using embeddings trained offline on a large
corpus of LLVM-IR. It is described in:

    Ben-Nun, T., Jakobovits, A. S., & Hoefler, T. (2018).
    `Neural code comprehension: A learnable representation of code semantics <https://papers.nips.cc/paper/2018/file/17c3433fecc21b57000debdf7ad5c930-Paper.pdf>`_.
    In Advances in Neural Information Processing Systems (pp. 3585-3597).

The inst2vec methodology comprises three steps, all of which are exposed as
observation spaces:

**Step 1: pre-processing**

The LLVM-IR statements are pre-processed to remove literals, identifiers, and
simplify the expressions. Using the Inst2vecPreprocessedText observation space
returns a list of pre-processed strings, one per statement. It could be useful
if you want to normalize the IR but then do your own embedding.

    >>> env.observation["Inst2vecPreprocessedText"]
    ['opaque = type opaque', ..., 'ret i32 <%ID>']

**Step 2: encoding**

Each of the pre-processed statements is mapped to an index into a vocabulary of
over 8k LLVM-IR statements. If a statement is not found in the vocabulary, it
maps to a special !UNK vocabulary item. Using the Inst2vecEmbeddingIndices
observation space returns a list of vocabulary indices. This would be useful if
you want to learn your own embeddings using the same vocabulary, or if you want
to use the inst2vec pre-trained embeddings but are processing them on a GPU
where you have already allocated and copied the embedding table, minimizing
transfer sizes.

    >>> env.observation["Inst2vecEmbeddingIndices"]
    [8564, 8564, 5, 46, ..., 257]

**Step 3: embedding**

The vocabulary indices are mapped to 200-D embedding vectors, producing an
np.array of shape (num_statements, 200). This could be fed into an LSTM to
produce a program embedding.

    >>> env.observation["Inst2vec"]
    array([[-0.26956588,  0.47407162, -0.36637706, ..., -0.49256894,
             0.8016193 ,  0.71160674],
           [-0.59749085,  0.63315004, -0.0308373 , ...,  0.14833118,
             0.86420786,  0.44808227],
           [-0.59749085,  0.63315004, -0.0308373 , ...,  0.14833118,
             0.86420786,  0.44808227],
           ...,
           [-0.37584195,  0.43671703, -0.5360456 , ...,  0.6030259 ,
             0.82574934,  0.6306344 ],
           [-0.59749085,  0.63315004, -0.0308373 , ...,  0.14833118,
             0.86420786,  0.44808227],
           [-0.43074277,  0.8589559 , -0.35770646, ...,  0.28785184,
             0.8492773 ,  0.8914213 ]], dtype=float32)

ProGraML
~~~~~~~~

+--------------------------+------------------------------------------------------+
| Observation space        | Shape                                                |
+==========================+======================================================+
| Programl                 | `str_list<>[0,inf]) -> json://networkx/MultiDiGraph` |
+--------------------------+------------------------------------------------------+

The ProGraML representation is a graph-based representation of LLVM-IR which
includes control-flow, data-flow, and call-flow. This graph is represented as
an `nx.MultiDiGraph <https://networkx.org/documentation/stable/reference/classes/multidigraph.html>`_.
ProGraML is described in:

    Cummins, C., Fisches, Z. V., Ben-Nun, T., Hoefler, T., & Leather, H. (2020).
    `ProGraML: Graph-based Deep Learning for Program Optimization and Analysis <https://arxiv.org/pdf/2003.10536.pdf>`_.
    arXiv preprint arXiv:2003.10536.

Example usage:

    >>> G = env.observation["Programl"]
    >>> G
    <networkx.classes.multidigraph.MultiDiGraph object at 0x7f9d8050ffa0>
    >>> G.number_of_nodes()
    6326
    >>> G.nodes[1000]
    {'block': 8, 'features': {'full_text': ['%439 = load double, double* @tmp2, align 8']}, 'function': 0, 'text': 'load', 'type': 0}
    >>> G.edge[0, 1, 0]
    {'flow': 2, 'position': 0}


Hardware Information
~~~~~~~~~~~~~~~~~~~~

+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Observation space    | Shape                                                                                                                                                                                                                                                   |
+======================+=========================================================================================================================================================================================================================================================+
| CpuInfo              | `Dict(cores_count:int, l1d_cache_count:int, l1d_cache_size:int, l1i_cache_count:int, l1i_cache_size:int, l2_cache_count:int, l2_cache_size:int, l3_cache_count:int, l3_cache_size:int, l4_cache_count:int, l4_cache_size:int, name:str_list<>[0,inf]))` |
+----------------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Essential performance information about the host CPU can be accessed as JSON
dictionary, extracted using the `cpuinfo <https://github.com/pytorch/cpuinfo>`_
library.

Example usage:

    >>> env.observation["CpuInfo"]
    {'cores_count': 8, 'l1d_cache_count': 8, ...}


Cost Models
~~~~~~~~~~~

+--------------------------+------------------------------------------------------------------------------------+
| Observation space        | Shape                                                                              |
+==========================+====================================================================================+
| IrInstructionCount       | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+
| IrInstructionCountO0     | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+
| IrInstructionCountO3     | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+
| IrInstructionCountOz     | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+
| ObjectTextSizeBytes      | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+
| ObjectTextSizeO0         | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+
| ObjectTextSizeO3         | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+
| ObjectTextSizeOz         | `Box(0, 9223372036854775807, (1,), int64)`                                         |
+--------------------------+------------------------------------------------------------------------------------+

Raw values from the cost models used to compute :ref:`rewards <reward>`.


.. _reward:

Reward Spaces
-------------

The goal of CompilerGym tasks is to minimize a cost function :math:`C(s)` which
takes as input the current program state :math:`s` and produces a real-valued
cost. At a given timestep, reward is the reduction in cost from the previous
state :math:`s_{t-1}` to the current state :math:`s_t`:

.. math::
    R(s_t) = C(s_{t-1}) - C(s_t)

Reward can be normalized using the cost of the program before any optimizations
are applied as the scaling factor:

.. math::
    R(s_t) = \frac{C(s_{t-1}) - C(s_t)}{C(s_{t=0})}

Normalized rewards are indicated by a :code:`Norm` suffix on the reward space
name.

Alternatively, rewards can be normalized by comparison to a baseline policy. The
baseline policies are derived from existing
`LLVM optimization levels <https://clang.llvm.org/docs/CommandGuide/clang.html#code-generation-options>`_:
:code:`-O3`, and :code:`-Oz`. When a baseline policy is used, reward is the
reduction in cost from the previous state, scaled by the *reduction in cost*
achieved by applying the baseline policy to produce a baseline state
:math:`s_b`:

.. math::
    R(s_t) = \frac{C(s_{t-1}) - C(s_t)}{{C(s_{t=0})} - C(s_b)}

These reward spaces are indicated by the baseline policy name as a suffix, e.g.
the reward space :code:`IrInstructionCountO3` is :code:`IrInstructionCount`
reward normalized to the :code:`-O3` baseline policy.


IR Instruction Count
~~~~~~~~~~~~~~~~~~~~

+------------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| Reward space           | Baseline Policy | Range       |   Success Threshold | Deterministic?   | Platform dependent?   |
+========================+=================+=============+=====================+==================+=======================+
| IrInstructionCount     |                 | (-inf, inf) |                     | Yes              | No                    |
+------------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| IrInstructionCountNorm |                 | (-inf, 1.0) |                     | Yes              | No                    |
+------------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| IrInstructionCountO3   | :code:`-O3`     | (-inf, inf) |                 1.0 | Yes              | No                    |
+------------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| IrInstructionCountOz   | :code:`-Oz`     | (-inf, inf) |                 1.0 | Yes              | No                    |
+------------------------+-----------------+-------------+---------------------+------------------+-----------------------+

The number of LLVM-IR instructions in the program can be used as a reward
signal either using the raw change in instruction count
(:code:`IrInstructionCount`), or by scaling the changes in instruction count
to the improvement made by the baseline :code:`-O3` or :code:`-Oz` LLVM
pipelines. LLVM-IR instruction count is fast to evaluate, deterministic, and
platform-independent, but is not a measure of true codesize reduction as it does
not take into account the effects of lowering.


Codesize
~~~~~~~~

+----------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| Reward space         | Baseline Policy | Range       |   Success Threshold | Deterministic?   | Platform dependent?   |
+======================+=================+=============+=====================+==================+=======================+
| ObjectTextSizeBytes  |                 | (-inf, inf) |                     | Yes              | Yes                   |
+----------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| ObjectTextSizeNorm   |                 | (-inf, 1.0) |                     | Yes              | Yes                   |
+----------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| ObjectTextSizeO3     | :code:`-O3`     | (-inf, inf) |                 1.0 | Yes              | Yes                   |
+----------------------+-----------------+-------------+---------------------+------------------+-----------------------+
| ObjectTextSizeOz     | :code:`-Oz`     | (-inf, inf) |                 1.0 | Yes              | Yes                   |
+----------------------+-----------------+-------------+---------------------+------------------+-----------------------+

The :code:`ObjectTextSizeBytes` reward signal returns the size of the
:code:`.TEXT` section of the module after lowering to an object file, before
linking. This is more expensive to compute than :code:`IrInstructionCount`. The
object file code size depends on the target platform, see
:func:`CompilerEnv.compiler_version <compiler_gym.envs.CompilerEnv.compiler_version>`.


Action Space
------------

The LLVM action space exposes the selection of semantics-preserving optimization
transforms as a discrete space.

+-----------------------------------+------------------------------------------------------------------------------+
| Action                            | Description                                                                  |
+===================================+==============================================================================+
| `-add-discriminators`             | Add DWARF path discriminators                                                |
+-----------------------------------+------------------------------------------------------------------------------+
| `-adce`                           | Aggressive Dead Code Elimination                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-aggressive-instcombine`         | Combine pattern based expressions                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-alignment-from-assumptions`     | Alignment from assumptions                                                   |
+-----------------------------------+------------------------------------------------------------------------------+
| `-always-inline`                  | Inliner for always_inline functions                                          |
+-----------------------------------+------------------------------------------------------------------------------+
| `-argpromotion`                   | Promote 'by reference' arguments to scalars                                  |
+-----------------------------------+------------------------------------------------------------------------------+
| `-attributor`                     | Deduce and propagate attributes                                              |
+-----------------------------------+------------------------------------------------------------------------------+
| `-barrier`                        | A No-Op Barrier Pass                                                         |
+-----------------------------------+------------------------------------------------------------------------------+
| `-bdce`                           | Bit-Tracking Dead Code Elimination                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-break-crit-edges`               | Break critical edges in CFG                                                  |
+-----------------------------------+------------------------------------------------------------------------------+
| `-simplifycfg`                    | Simplify the CFG                                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-callsite-splitting`             | Call-site splitting                                                          |
+-----------------------------------+------------------------------------------------------------------------------+
| `-called-value-propagation`       | Called Value Propagation                                                     |
+-----------------------------------+------------------------------------------------------------------------------+
| `-canonicalize-aliases`           | Canonicalize aliases                                                         |
+-----------------------------------+------------------------------------------------------------------------------+
| `-consthoist`                     | Constant Hoisting                                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-constmerge`                     | Merge Duplicate Global Constants                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-constprop`                      | Simple constant propagation                                                  |
+-----------------------------------+------------------------------------------------------------------------------+
| `-coro-cleanup`                   | Lower all coroutine related intrinsics                                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-coro-early`                     | Lower early coroutine intrinsics                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-coro-elide`                     | Coroutine frame allocation elision and indirect calls replacement            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-coro-split`                     | Split coroutine into a set of functions driving its state machine            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-correlated-propagation`         | Value Propagation                                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-cross-dso-cfi`                  | Cross-DSO CFI                                                                |
+-----------------------------------+------------------------------------------------------------------------------+
| `-deadargelim`                    | Dead Argument Elimination                                                    |
+-----------------------------------+------------------------------------------------------------------------------+
| `-dce`                            | Dead Code Elimination                                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-die`                            | Dead Instruction Elimination                                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-dse`                            | Dead Store Elimination                                                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-reg2mem`                        | Demote all values to stack slots                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-div-rem-pairs`                  | Hoist/decompose integer division and remainder                               |
+-----------------------------------+------------------------------------------------------------------------------+
| `-early-cse-memssa`               | Early CSE w/ MemorySSA                                                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-elim-avail-extern`              | Eliminate Available Externally Globals                                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-ee-instrument`                  | Instrument function entry/exit with calls to e.g. mcount()(pre inlining)     |
+-----------------------------------+------------------------------------------------------------------------------+
| `-flattencfg`                     | Flatten the CFG                                                              |
+-----------------------------------+------------------------------------------------------------------------------+
| `-float2int`                      | Float to int                                                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-forceattrs`                     | Force set function attributes                                                |
+-----------------------------------+------------------------------------------------------------------------------+
| `-inline`                         | Function Integration/Inlining                                                |
+-----------------------------------+------------------------------------------------------------------------------+
| `-insert-gcov-profiling`          | Insert instrumentation for GCOV profiling                                    |
+-----------------------------------+------------------------------------------------------------------------------+
| `-gvn-hoist`                      | Early GVN Hoisting of Expressions                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-gvn`                            | Global Value Numbering                                                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-globaldce`                      | Dead Global Elimination                                                      |
+-----------------------------------+------------------------------------------------------------------------------+
| `-globalopt`                      | Global Variable Optimizer                                                    |
+-----------------------------------+------------------------------------------------------------------------------+
| `-globalsplit`                    | Global splitter                                                              |
+-----------------------------------+------------------------------------------------------------------------------+
| `-guard-widening`                 | Widen guards                                                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-hotcoldsplit`                   | Hot Cold Splitting                                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-ipconstprop`                    | Interprocedural constant propagation                                         |
+-----------------------------------+------------------------------------------------------------------------------+
| `-ipsccp`                         | Interprocedural Sparse Conditional Constant Propagation                      |
+-----------------------------------+------------------------------------------------------------------------------+
| `-indvars`                        | Induction Variable Simplification                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-irce`                           | Inductive range check elimination                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-infer-address-spaces`           | Infer address spaces                                                         |
+-----------------------------------+------------------------------------------------------------------------------+
| `-inferattrs`                     | Infer set function attributes                                                |
+-----------------------------------+------------------------------------------------------------------------------+
| `-inject-tli-mappings`            | Inject TLI Mappings                                                          |
+-----------------------------------+------------------------------------------------------------------------------+
| `-instsimplify`                   | Remove redundant instructions                                                |
+-----------------------------------+------------------------------------------------------------------------------+
| `-instcombine`                    | Combine redundant instructions                                               |
+-----------------------------------+------------------------------------------------------------------------------+
| `-instnamer`                      | Assign names to anonymous instructions                                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-jump-threading`                 | Jump Threading                                                               |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lcssa`                          | Loop-Closed SSA Form Pass                                                    |
+-----------------------------------+------------------------------------------------------------------------------+
| `-licm`                           | Loop Invariant Code Motion                                                   |
+-----------------------------------+------------------------------------------------------------------------------+
| `-libcalls-shrinkwrap`            | Conditionally eliminate dead library calls                                   |
+-----------------------------------+------------------------------------------------------------------------------+
| `-load-store-vectorizer`          | Vectorize load and Store instructions                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-data-prefetch`             | Loop Data Prefetch                                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-deletion`                  | Delete dead loops                                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-distribute`                | Loop Distribution                                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-fusion`                    | Loop Fusion                                                                  |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-guard-widening`            | Widen guards (within a single loop, as a loop pass)                          |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-idiom`                     | Recognize loop idioms                                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-instsimplify`              | Simplify instructions in loops                                               |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-interchange`               | Interchanges loops for cache reuse                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-load-elim`                 | Loop Load Elimination                                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-predication`               | Loop predication                                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-reroll`                    | Reroll loops                                                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-rotate`                    | Rotate Loops                                                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-simplifycfg`               | Simplify loop CFG                                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-simplify`                  | Canonicalize natural loops                                                   |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-sink`                      | Loop Sink                                                                    |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-reduce`                    | Loop Strength Reduction                                                      |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-unroll-and-jam`            | Unroll and Jam loops                                                         |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-unroll`                    | Unroll loops                                                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-unswitch`                  | Unswitch loops                                                               |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-vectorize`                 | Loop Vectorization                                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-versioning-licm`           | Loop Versioning For LICM                                                     |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loop-versioning`                | Loop Versioning                                                              |
+-----------------------------------+------------------------------------------------------------------------------+
| `-loweratomic`                    | Lower atomic intrinsics to non-atomic form                                   |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lower-constant-intrinsics`      | Lower constant intrinsics                                                    |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lower-expect`                   | Lower 'expect' Intrinsics                                                    |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lower-guard-intrinsic`          | Lower the guard intrinsic to normal control flow                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lowerinvoke`                    | Lower invoke and unwind, for unwindless code generators                      |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lower-matrix-intrinsics`        | Lower the matrix intrinsics                                                  |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lowerswitch`                    | Lower SwitchInst's to branches                                               |
+-----------------------------------+------------------------------------------------------------------------------+
| `-lower-widenable-condition`      | Lower the widenable condition to default true value                          |
+-----------------------------------+------------------------------------------------------------------------------+
| `-memcpyopt`                      | MemCpy Optimization                                                          |
+-----------------------------------+------------------------------------------------------------------------------+
| `-mergefunc`                      | Merge Functions                                                              |
+-----------------------------------+------------------------------------------------------------------------------+
| `-mergeicmps`                     | Merge contiguous icmps into a memcmp                                         |
+-----------------------------------+------------------------------------------------------------------------------+
| `-mldst-motion`                   | MergedLoadStoreMotion                                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-sancov`                         | Pass for instrumenting coverage on functions                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-name-anon-globals`              | Provide a name to nameless globals                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-nary-reassociate`               | Nary reassociation                                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-newgvn`                         | Global Value Numbering                                                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-pgo-memop-opt`                  | Optimize memory intrinsic using its size value profile                       |
+-----------------------------------+------------------------------------------------------------------------------+
| `-partial-inliner`                | Partial Inliner                                                              |
+-----------------------------------+------------------------------------------------------------------------------+
| `-partially-inline-libcalls`      | Partially inline calls to library functions                                  |
+-----------------------------------+------------------------------------------------------------------------------+
| `-post-inline-ee-instrument`      | Instrument function entry/exit with calls to e.g. mcount()" "(post inlining) |
+-----------------------------------+------------------------------------------------------------------------------+
| `-functionattrs`                  | Deduce function attributes                                                   |
+-----------------------------------+------------------------------------------------------------------------------+
| `-mem2reg`                        | Promote Memory to " "Register                                                |
+-----------------------------------+------------------------------------------------------------------------------+
| `-prune-eh`                       | Remove unused exception handling info                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-reassociate`                    | Reassociate expressions                                                      |
+-----------------------------------+------------------------------------------------------------------------------+
| `-redundant-dbg-inst-elim`        | Redundant Dbg Instruction Elimination                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-rpo-functionattrs`              | Deduce function attributes in RPO                                            |
+-----------------------------------+------------------------------------------------------------------------------+
| `-rewrite-statepoints-for-gc`     | Make relocations explicit at statepoints                                     |
+-----------------------------------+------------------------------------------------------------------------------+
| `-sccp`                           | Sparse Conditional Constant Propagation                                      |
+-----------------------------------+------------------------------------------------------------------------------+
| `-slp-vectorizer`                 | SLP Vectorizer                                                               |
+-----------------------------------+------------------------------------------------------------------------------+
| `-sroa`                           | Scalar Replacement Of Aggregates                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-scalarizer`                     | Scalarize vector operations                                                  |
+-----------------------------------+------------------------------------------------------------------------------+
| `-separate-const-offset-from-gep` | Split GEPs to a variadic base and a constant offset for better CSE           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-simple-loop-unswitch`           | Simple unswitch loops                                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-sink`                           | Code sinking                                                                 |
+-----------------------------------+------------------------------------------------------------------------------+
| `-speculative-execution`          | Speculatively execute instructions                                           |
+-----------------------------------+------------------------------------------------------------------------------+
| `-slsr`                           | Straight line strength reduction                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-strip-dead-prototypes`          | Strip Unused Function Prototypes                                             |
+-----------------------------------+------------------------------------------------------------------------------+
| `-strip-debug-declare`            | Strip all llvm.dbg.declare intrinsics                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-strip-nondebug`                 | Strip all symbols, except dbg symbols, from a module                         |
+-----------------------------------+------------------------------------------------------------------------------+
| `-strip`                          | Strip all symbols from a module                                              |
+-----------------------------------+------------------------------------------------------------------------------+
| `-tailcallelim`                   | Tail Call Elimination                                                        |
+-----------------------------------+------------------------------------------------------------------------------+
| `-mergereturn`                    | Unify function exit nodes                                                    |
+-----------------------------------+------------------------------------------------------------------------------+

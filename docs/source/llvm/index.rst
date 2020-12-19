LLVM Environments Reference
===========================

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

+------------------------+---------------------------------+-----------------+----------------+
| Dataset                | License                         | Num. Benchmarks | Size on disk   |
+========================+=================================+=================+================+
| blas-v0                | BSD 3-Clause                    | 300             | 4.0 MB         |
+------------------------+---------------------------------+-----------------+----------------+
| cBench-v0              | BSD 3-Clause                    | 23              | 7.2 MB         |
+------------------------+---------------------------------+-----------------+----------------+
| github-v0              | CC BY 4.0                       | 50,708          | 726.0 MB       |
+------------------------+---------------------------------+-----------------+----------------+
| linux-v0               | GPL-2.0                         | 13,920          | 516.0 MB       |
+------------------------+---------------------------------+-----------------+----------------+
| mibench-v0             | BSD 3-Clause                    | 40              | 238.5 kB       |
+------------------------+---------------------------------+-----------------+----------------+
| npb-v0                 | NASA Open Source Agreement v1.3 | 122             | 2.3 MB         |
+------------------------+---------------------------------+-----------------+----------------+
| opencv-v0              | Apache 2.0                      | 442             | 21.9 MB        |
+------------------------+---------------------------------+-----------------+----------------+
| poj104-v0              | BSD 3-Clause                    | 49,628          | 304.2 MB       |
+------------------------+---------------------------------+-----------------+----------------+
| polybench-v0           | BSD 3-Clause                    | 27              | 162.6 kB       |
+------------------------+---------------------------------+-----------------+----------------+
| tensorflow-v0          | Apache 2.0                      | 1,985           | 299.7 MB       |
+------------------------+---------------------------------+-----------------+----------------+

Install these datasets using the :mod:`compiler_gym.bin.datasets` command line
tool, or programatically using
:meth:`CompilerEnv.require_datasets() <compiler_gym.envs.CompilerEnv.require_datasets>`:

    >>> env = gym.make("llvm-v0")
    >>> env.require_datasets(["tensorflow-v0", "npb-v0"])


Observation Spaces
------------------

We provide several observation spaces for LLVM based on published compiler
research.

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

+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Observation space        | Shape                                                                                                                                                                                                                                                                                                                                                                 |
+==========================+=======================================================================================================================================================================================================================================================================================================================================================================+
| CpuInfo                  | `str_list<>[0,inf]) -> json://`                                                                                                                                                                                                                                                                                                                                       |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| CpuInfoDict              | `Dict(cores_count:int<-inf,inf>, l1d_cache_count:int<-inf,inf>, l1d_cache_size:int<-inf,inf>, l1i_cache_count:int<-inf,inf>, l1i_cache_size:int<-inf,inf>, l2_cache_count:int<-inf,inf>, l2_cache_size:int<-inf,inf>, l3_cache_count:int<-inf,inf>, l3_cache_size:int<-inf,inf>, l4_cache_count:int<-inf,inf>, l4_cache_size:int<-inf,inf>, name:str_list<>[0,inf]))` |
+--------------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Essential performance information about the host CPU can be accessed as JSON
dictionary, extracted using the `cpuinfo <https://github.com/pytorch/cpuinfo>`_
library.

Example usage:

    >>> env.observation["CpuInfo"]
    {'cores_count': 8, 'l1d_cache_count': 8, ...}

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

Note that the files generated by the :code:`BitcodeFile` observation space are
put in a temporary directory that is removed when :code:`env.close()` is called.

Reward Spaces
-------------

Codesize
~~~~~~~~

+--------------------------+-------------+
| Reward space             | Range       |
+==========================+=============+
| IrInstructionCount       | (-inf, 0.0) |
+--------------------------+-------------+
| IrInstructionCountO3     | (0.0, inf)  |
+--------------------------+-------------+
| IrInstructionCountOz     | (0.0, inf)  |
+--------------------------+-------------+
| IrInstructionCountOzDiff | (-inf, inf) |
+--------------------------+-------------+
| NativeTextSizeBytes      | (-inf, 0.0) |
+--------------------------+-------------+

The number of LLVM-IR instructions in the program can be used as reward
signals, either using the raw instruction count (:code:`IrInstructionCount`),
or by normalizing the instruction count relative to the instruction count when
the program is optimized using the :code:`-O3` of :code:`-Oz` LLVM pipelines.
LLVM-IR instruction count is fast to evaluate, deterministic, and
platform-independent, but is not a measure of true codesize reduction as it does
not take into account the effects of lowering.

The :code:`NativeTextSizeBytes` reward signal returns the size of the
:code:`.TEXT` section of the module after lowering to native code. This is more
expensive to compute than :code:`IrInstructionCount`. The native code size
depends on the target platform.


Action Space
------------

The LLVM action space exposes the selection of semantics-preserving optimization
transforms as a discrete space.

+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Action                                | Flag                              | Description                                                                  |
+=======================================+===================================+==============================================================================+
| AddDiscriminatorsPass                 | `-add-discriminators`             | Add DWARF path discriminators                                                |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| AggressiveDcepass                     | `-adce`                           | Aggressive Dead Code Elimination                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| AggressiveInstCombinerPass            | `-aggressive-instcombine`         | Combine pattern based expressions                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| AlignmentFromAssumptionsPass          | `-alignment-from-assumptions`     | Alignment from assumptions                                                   |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| AlwaysInlinerLegacyPass               | `-always-inline`                  | Inliner for always_inline functions                                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ArgumentPromotionPass                 | `-argpromotion`                   | Promote 'by reference' arguments to scalars                                  |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| AttributorLegacyPass                  | `-attributor`                     | Deduce and propagate attributes                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| BarrierNoopPass                       | `-barrier`                        | A No-Op Barrier Pass                                                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| BitTrackingDcepass                    | `-bdce`                           | Bit-Tracking Dead Code Elimination                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| BlockExtractorPass                    | `-extract-blocks`                 | Extract basic blocks from module                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| BoundsCheckingLegacyPass              | `-bounds-checking`                | Run-time bounds checking                                                     |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| BreakCriticalEdgesPass                | `-break-crit-edges`               | Break critical edges in CFG                                                  |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CfgsimplificationPass                 | `-simplifycfg`                    | Simplify the CFG                                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CallSiteSplittingPass                 | `-callsite-splitting`             | Call-site splitting                                                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CalledValuePropagationPass            | `-called-value-propagation`       | Called Value Propagation                                                     |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CanonicalizeAliasesPass               | `-canonicalize-aliases`           | Canonicalize aliases                                                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ConstantHoistingPass                  | `-consthoist`                     | Constant Hoisting                                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ConstantMergePass                     | `-constmerge`                     | Merge Duplicate Global Constants                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ConstantPropagationPass               | `-constprop`                      | Simple constant propagation                                                  |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ControlHeightReductionLegacyPass      | `-chr`                            | Reduce control height in the hot paths                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CoroCleanupLegacyPass                 | `-coro-cleanup`                   | Lower all coroutine related intrinsics                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CoroEarlyLegacyPass                   | `-coro-early`                     | Lower early coroutine intrinsics                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CoroElideLegacyPass                   | `-coro-elide`                     | Coroutine frame allocation elision and indirect calls replacement            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CoroSplitLegacyPass                   | `-coro-split`                     | Split coroutine into a set of functions driving its state machine            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CorrelatedValuePropagationPass        | `-correlated-propagation`         | Value Propagation                                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| CrossDsocfipass                       | `-cross-dso-cfi`                  | Cross-DSO CFI                                                                |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| DeadArgEliminationPass                | `-deadargelim`                    | Dead Argument Elimination                                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| DeadCodeEliminationPass               | `-dce`                            | Dead Code Elimination                                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| DeadInstEliminationPass               | `-die`                            | Dead Instruction Elimination                                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| DeadStoreEliminationPass              | `-dse`                            | Dead Store Elimination                                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| DemoteRegisterToMemoryPass            | `-reg2mem`                        | Demote all values to stack slots                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| DivRemPairsPass                       | `-div-rem-pairs`                  | Hoist/decompose integer division and remainder                               |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| EarlyCsepass                          | `-early-cse-memssa`               | Early CSE w/ MemorySSA                                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| EliminateAvailableExternallyPass      | `-elim-avail-extern`              | Eliminate Available Externally Globals                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| EntryExitInstrumenterPass             | `-ee-instrument`                  | Instrument function entry/exit with calls to e.g. mcount()(pre inlining)     |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| FlattenCfgpass                        | `-flattencfg`                     | Flatten the CFG                                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Float2intPass                         | `-float2int`                      | Float to int                                                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ForceFunctionAttrsLegacyPass          | `-forceattrs`                     | Force set function attributes                                                |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| FunctionInliningPass                  | `-inline`                         | Function Integration/Inlining                                                |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| GcovprofilerPass                      | `-insert-gcov-profiling`          | Insert instrumentation for GCOV profiling                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| GvnhoistPass                          | `-gvn-hoist`                      | Early GVN Hoisting of Expressions                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Gvnpass                               | `-gvn`                            | Global Value Numbering                                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| GvnsinkPass                           | `-gvn-sink`                       | Early GVN sinking of Expressions                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| GlobalDcepass                         | `-globaldce`                      | Dead Global Elimination                                                      |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| GlobalOptimizerPass                   | `-globalopt`                      | Global Variable Optimizer                                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| GlobalSplitPass                       | `-globalsplit`                    | Global splitter                                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| GuardWideningPass                     | `-guard-widening`                 | Widen guards                                                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| HotColdSplittingPass                  | `-hotcoldsplit`                   | Hot Cold Splitting                                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| IpconstantPropagationPass             | `-ipconstprop`                    | Interprocedural constant propagation                                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Ipsccppass                            | `-ipsccp`                         | Interprocedural Sparse Conditional Constant Propagation                      |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| IndVarSimplifyPass                    | `-indvars`                        | Induction Variable Simplification                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| InductiveRangeCheckEliminationPass    | `-irce`                           | Inductive range check elimination                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| InferAddressSpacesPass                | `-infer-address-spaces`           | Infer address spaces                                                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| InferFunctionAttrsLegacyPass          | `-inferattrs`                     | Infer set function attributes                                                |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| InjectTlimappingsLegacyPass           | `-inject-tli-mappings`            | Inject TLI Mappings                                                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| InstSimplifyLegacyPass                | `-instsimplify`                   | Remove redundant instructions                                                |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| InstructionCombiningPass              | `-instcombine`                    | Combine redundant instructions                                               |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| InstructionNamerPass                  | `-instnamer`                      | Assign names to anonymous instructions                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| JumpThreadingPass                     | `-jump-threading`                 | Jump Threading                                                               |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Lcssapass                             | `-lcssa`                          | Loop-Closed SSA Form Pass                                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Licmpass                              | `-licm`                           | Loop Invariant Code Motion                                                   |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LibCallsShrinkWrapPass                | `-libcalls-shrinkwrap`            | Conditionally eliminate dead library calls                                   |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoadStoreVectorizerPass               | `-load-store-vectorizer`          | Vectorize load and Store instructions                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopDataPrefetchPass                  | `-loop-data-prefetch`             | Loop Data Prefetch                                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopDeletionPass                      | `-loop-deletion`                  | Delete dead loops                                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopDistributePass                    | `-loop-distribute`                | Loop Distribution                                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopExtractorPass                     | `-loop-extract`                   | Extract loops into new functions                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopFusePass                          | `-loop-fusion`                    | Loop Fusion                                                                  |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopGuardWideningPass                 | `-loop-guard-widening`            | Widen guards (within a single loop, as a loop pass)                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopIdiomPass                         | `-loop-idiom`                     | Recognize loop idioms                                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopInstSimplifyPass                  | `-loop-instsimplify`              | Simplify instructions in loops                                               |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopInterchangePass                   | `-loop-interchange`               | Interchanges loops for cache reuse                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopLoadEliminationPass               | `-loop-load-elim`                 | Loop Load Elimination                                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopPredicationPass                   | `-loop-predication`               | Loop predication                                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopRerollPass                        | `-loop-reroll`                    | Reroll loops                                                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopRotatePass                        | `-loop-rotate`                    | Rotate Loops                                                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopSimplifyCfgpass                   | `-loop-simplifycfg`               | Simplify loop CFG                                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopSimplifyPass                      | `-loop-simplify`                  | Canonicalize natural loops                                                   |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopSinkPass                          | `-loop-sink`                      | Loop Sink                                                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopStrengthReducePass                | `-loop-reduce`                    | Loop Strength Reduction                                                      |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopUnrollAndJamPass                  | `-loop-unroll-and-jam`            | Unroll and Jam loops                                                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopUnrollPass                        | `-loop-unroll`                    | Unroll loops                                                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopUnswitchPass                      | `-loop-unswitch`                  | Unswitch loops                                                               |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopVectorizePass                     | `-loop-vectorize`                 | Loop Vectorization                                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopVersioningLicmpass                | `-loop-versioning-licm`           | Loop Versioning For LICM                                                     |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LoopVersioningPass                    | `-loop-versioning`                | Loop Versioning                                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerAtomicPass                       | `-loweratomic`                    | Lower atomic intrinsics to non-atomic form                                   |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerConstantIntrinsicsPass           | `-lower-constant-intrinsics`      | Lower constant intrinsics                                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerExpectIntrinsicPass              | `-lower-expect`                   | Lower 'expect' Intrinsics                                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerGuardIntrinsicPass               | `-lower-guard-intrinsic`          | Lower the guard intrinsic to normal control flow                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerInvokePass                       | `-lowerinvoke`                    | Lower invoke and unwind, for unwindless code generators                      |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerMatrixIntrinsicsPass             | `-lower-matrix-intrinsics`        | Lower the matrix intrinsics                                                  |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerSwitchPass                       | `-lowerswitch`                    | Lower SwitchInst's to branches                                               |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| LowerWidenableConditionPass           | `-lower-widenable-condition`      | Lower the widenable condition to default true value                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| MemCpyOptPass                         | `-memcpyopt`                      | MemCpy Optimization                                                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| MergeFunctionsPass                    | `-mergefunc`                      | Merge Functions                                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| MergeIcmpsLegacyPass                  | `-mergeicmps`                     | Merge contiguous icmps into a memcmp                                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| MergedLoadStoreMotionPass             | `-mldst-motion`                   | MergedLoadStoreMotion                                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ModuleSanitizerCoverageLegacyPassPass | `-sancov`                         | Pass for instrumenting coverage on functions                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| NameAnonGlobalPass                    | `-name-anon-globals`              | Provide a name to nameless globals                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| NaryReassociatePass                   | `-nary-reassociate`               | Nary reassociation                                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| NewGvnpass                            | `-newgvn`                         | Global Value Numbering                                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ObjCarcapelimPass                     | `-objc-arc-apelim`                | ObjC ARC autorelease pool elimination                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ObjCarccontractPass                   | `-objc-arc-contract`              | ObjC ARC contraction                                                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ObjCarcexpandPass                     | `-objc-arc-expand`                | ObjC ARC expansion                                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ObjCarcoptPass                        | `-objc-arc`                       | ObjC ARC optimization                                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PgomemOpsizeOptLegacyPass             | `-pgo-memop-opt`                  | Optimize memory intrinsic using its size value profile                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PartialInliningPass                   | `-partial-inliner`                | Partial Inliner                                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PartiallyInlineLibCallsPass           | `-partially-inline-libcalls`      | Partially inline calls to library functions                                  |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PlaceSafepointsPass                   | `-place-safepoints`               | Place Safepoints                                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PostInlineEntryExitInstrumenterPass   | `-post-inline-ee-instrument`      | Instrument function entry/exit with calls to e.g. mcount()" "(post inlining) |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PostOrderFunctionAttrsLegacyPass      | `-functionattrs`                  | Deduce function attributes                                                   |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PromoteMemoryToRegisterPass           | `-mem2reg`                        | Promote Memory to " "Register                                                |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| PruneEhpass                           | `-prune-eh`                       | Remove unused exception handling info                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ReassociatePass                       | `-reassociate`                    | Reassociate expressions                                                      |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| RedundantDbgInstEliminationPass       | `-redundant-dbg-inst-elim`        | Redundant Dbg Instruction Elimination                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ReversePostOrderFunctionAttrsPass     | `-rpo-functionattrs`              | Deduce function attributes in RPO                                            |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| RewriteStatepointsForGclegacyPass     | `-rewrite-statepoints-for-gc`     | Make relocations explicit at statepoints                                     |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| RewriteSymbolsPass                    | `-rewrite-symbols`                | Rewrite Symbols                                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Sccppass                              | `-sccp`                           | Sparse Conditional Constant Propagation                                      |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| SlpvectorizerPass                     | `-slp-vectorizer`                 | SLP Vectorizer                                                               |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| Sroapass                              | `-sroa`                           | Scalar Replacement Of Aggregates                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| ScalarizerPass                        | `-scalarizer`                     | Scalarize vector operations                                                  |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| SeparateConstOffsetFromGeppass        | `-separate-const-offset-from-gep` | Split GEPs to a variadic base and a constant offset for better CSE           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| SimpleLoopUnswitchLegacyPass          | `-simple-loop-unswitch`           | Simple unswitch loops                                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| SingleLoopExtractorPass               | `-loop-extract-single`            | Extract at most one loop into a new function                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| SinkingPass                           | `-sink`                           | Code sinking                                                                 |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| SpeculativeExecutionPass              | `-speculative-execution`          | Speculatively execute instructions                                           |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StraightLineStrengthReducePass        | `-slsr`                           | Straight line strength reduction                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StripDeadDebugInfoPass                | `-strip-dead-debug-info`          | Strip debug info for unused symbols                                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StripDeadPrototypesPass               | `-strip-dead-prototypes`          | Strip Unused Function Prototypes                                             |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StripDebugDeclarePass                 | `-strip-debug-declare`            | Strip all llvm.dbg.declare intrinsics                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StripNonDebugSymbolsPass              | `-strip-nondebug`                 | Strip all symbols, except dbg symbols, from a module                         |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StripNonLineTableDebugInfoPass        | `-strip-nonlinetable-debuginfo`   | Strip all debug info except linetables                                       |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StripSymbolsPass                      | `-strip`                          | Strip all symbols from a module                                              |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| StructurizeCfgpass                    | `-structurizecfg`                 | Structurize the CFG                                                          |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| TailCallEliminationPass               | `-tailcallelim`                   | Tail Call Elimination                                                        |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+
| UnifyFunctionExitNodesPass            | `-mergereturn`                    | Unify function exit nodes                                                    |
+---------------------------------------+-----------------------------------+------------------------------------------------------------------------------+

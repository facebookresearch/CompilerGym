/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

const actionsDict = [
  {
    Action: "-add-discriminators",
    Description: "Add DWARF path discriminators",
  },
  {
    Action: "-adce",
    Description: "Aggressive Dead Code Elimination",
  },
  {
    Action: "-aggressive-instcombine",
    Description: "Combine pattern based expressions",
  },
  {
    Action: "-alignment-from-assumptions",
    Description: "Alignment from assumptions",
  },
  {
    Action: "-always-inline",
    Description: "Inliner for always_inline functions",
  },
  {
    Action: "-argpromotion",
    Description: "Promote ‘by reference’ arguments to scalars",
  },
  {
    Action: "-attributor",
    Description: "Deduce and propagate attributes",
  },
  {
    Action: "-barrier",
    Description: "A No-Op Barrier Pass",
  },
  {
    Action: "-bdce",
    Description: "Bit-Tracking Dead Code Elimination",
  },
  {
    Action: "-break-crit-edges",
    Description: "Break critical edges in CFG",
  },
  {
    Action: "-simplifycfg",
    Description: "Simplify the CFG",
  },
  {
    Action: "-callsite-splitting",
    Description: "Call-site splitting",
  },
  {
    Action: "-called-value-propagation",
    Description: "Called Value Propagation",
  },
  {
    Action: "-canonicalize-aliases",
    Description: "Canonicalize aliases",
  },
  {
    Action: "-consthoist",
    Description: "Constant Hoisting",
  },
  {
    Action: "-constmerge",
    Description: "Merge Duplicate Global Constants",
  },
  {
    Action: "-constprop",
    Description: "Simple constant propagation",
  },
  {
    Action: "-coro-cleanup",
    Description: "Lower all coroutine related intrinsics",
  },
  {
    Action: "-coro-early",
    Description: "Lower early coroutine intrinsics",
  },
  {
    Action: "-coro-elide",
    Description:
      "Coroutine frame allocation elision and indirect calls replacement",
  },
  {
    Action: "-coro-split",
    Description:
      "Split coroutine into a set of functions driving its state machine",
  },
  {
    Action: "-correlated-propagation",
    Description: "Value Propagation",
  },
  {
    Action: "-cross-dso-cfi",
    Description: "Cross-DSO CFI",
  },
  {
    Action: "-deadargelim",
    Description: "Dead Argument Elimination",
  },
  {
    Action: "-dce",
    Description: "Dead Code Elimination",
  },
  {
    Action: "-die",
    Description: "Dead Instruction Elimination",
  },
  {
    Action: "-dse",
    Description: "Dead Store Elimination",
  },
  {
    Action: "-reg2mem",
    Description: "Demote all values to stack slots",
  },
  {
    Action: "-div-rem-pairs",
    Description: "Hoist/decompose integer division and remainder",
  },
  {
    Action: "-early-cse-memssa",
    Description: "Early CSE w/ MemorySSA",
  },
  {
    Action: "-elim-avail-extern",
    Description: "Eliminate Available Externally Globals",
  },
  {
    Action: "-ee-instrument",
    Description:
      "Instrument function entry/exit with calls to e.g. mcount()(pre inlining)",
  },
  {
    Action: "-flattencfg",
    Description: "Flatten the CFG",
  },
  {
    Action: "-float2int",
    Description: "Float to int",
  },
  {
    Action: "-forceattrs",
    Description: "Force set function attributes",
  },
  {
    Action: "-inline",
    Description: "Function Integration/Inlining",
  },
  {
    Action: "-insert-gcov-profiling",
    Description: "Insert instrumentation for GCOV profiling",
  },
  {
    Action: "-gvn-hoist",
    Description: "Early GVN Hoisting of Expressions",
  },
  {
    Action: "-gvn",
    Description: "Global Value Numbering",
  },
  {
    Action: "-globaldce",
    Description: "Dead Global Elimination",
  },
  {
    Action: "-globalopt",
    Description: "Global Variable Optimizer",
  },
  {
    Action: "-globalsplit",
    Description: "Global splitter",
  },
  {
    Action: "-guard-widening",
    Description: "Widen guards",
  },
  {
    Action: "-hotcoldsplit",
    Description: "Hot Cold Splitting",
  },
  {
    Action: "-ipconstprop",
    Description: "Interprocedural constant propagation",
  },
  {
    Action: "-ipsccp",
    Description: "Interprocedural Sparse Conditional Constant Propagation",
  },
  {
    Action: "-indvars",
    Description: "Induction Variable Simplification",
  },
  {
    Action: "-irce",
    Description: "Inductive range check elimination",
  },
  {
    Action: "-infer-address-spaces",
    Description: "Infer address spaces",
  },
  {
    Action: "-inferattrs",
    Description: "Infer set function attributes",
  },
  {
    Action: "-inject-tli-mappings",
    Description: "Inject TLI Mappings",
  },
  {
    Action: "-instsimplify",
    Description: "Remove redundant instructions",
  },
  {
    Action: "-instcombine",
    Description: "Combine redundant instructions",
  },
  {
    Action: "-instnamer",
    Description: "Assign names to anonymous instructions",
  },
  {
    Action: "-jump-threading",
    Description: "Jump Threading",
  },
  {
    Action: "-lcssa",
    Description: "Loop-Closed SSA Form Pass",
  },
  {
    Action: "-licm",
    Description: "Loop Invariant Code Motion",
  },
  {
    Action: "-libcalls-shrinkwrap",
    Description: "Conditionally eliminate dead library calls",
  },
  {
    Action: "-load-store-vectorizer",
    Description: "Vectorize load and Store instructions",
  },
  {
    Action: "-loop-data-prefetch",
    Description: "Loop Data Prefetch",
  },
  {
    Action: "-loop-deletion",
    Description: "Delete dead loops",
  },
  {
    Action: "-loop-distribute",
    Description: "Loop Distribution",
  },
  {
    Action: "-loop-fusion",
    Description: "Loop Fusion",
  },
  {
    Action: "-loop-guard-widening",
    Description: "Widen guards (within a single loop, as a loop pass)",
  },
  {
    Action: "-loop-idiom",
    Description: "Recognize loop idioms",
  },
  {
    Action: "-loop-instsimplify",
    Description: "Simplify instructions in loops",
  },
  {
    Action: "-loop-interchange",
    Description: "Interchanges loops for cache reuse",
  },
  {
    Action: "-loop-load-elim",
    Description: "Loop Load Elimination",
  },
  {
    Action: "-loop-predication",
    Description: "Loop predication",
  },
  {
    Action: "-loop-reroll",
    Description: "Reroll loops",
  },
  {
    Action: "-loop-rotate",
    Description: "Rotate Loops",
  },
  {
    Action: "-loop-simplifycfg",
    Description: "Simplify loop CFG",
  },
  {
    Action: "-loop-simplify",
    Description: "Canonicalize natural loops",
  },
  {
    Action: "-loop-sink",
    Description: "Loop Sink",
  },
  {
    Action: "-loop-reduce",
    Description: "Loop Strength Reduction",
  },
  {
    Action: "-loop-unroll-and-jam",
    Description: "Unroll and Jam loops",
  },
  {
    Action: "-loop-unroll",
    Description: "Unroll loops",
  },
  {
    Action: "-loop-unswitch",
    Description: "Unswitch loops",
  },
  {
    Action: "-loop-vectorize",
    Description: "Loop Vectorization",
  },
  {
    Action: "-loop-versioning-licm",
    Description: "Loop Versioning For LICM",
  },
  {
    Action: "-loop-versioning",
    Description: "Loop Versioning",
  },
  {
    Action: "-loweratomic",
    Description: "Lower atomic intrinsics to non-atomic form",
  },
  {
    Action: "-lower-constant-intrinsics",
    Description: "Lower constant intrinsics",
  },
  {
    Action: "-lower-expect",
    Description: "Lower ‘expect’ Intrinsics",
  },
  {
    Action: "-lower-guard-intrinsic",
    Description: "Lower the guard intrinsic to normal control flow",
  },
  {
    Action: "-lowerinvoke",
    Description: "Lower invoke and unwind, for unwindless code generators",
  },
  {
    Action: "-lower-matrix-intrinsics",
    Description: "Lower the matrix intrinsics",
  },
  {
    Action: "-lowerswitch",
    Description: "Lower SwitchInst’s to branches",
  },
  {
    Action: "-lower-widenable-condition",
    Description: "Lower the widenable condition to default true value",
  },
  {
    Action: "-memcpyopt",
    Description: "MemCpy Optimization",
  },
  {
    Action: "-mergefunc",
    Description: "Merge Functions",
  },
  {
    Action: "-mergeicmps",
    Description: "Merge contiguous icmps into a memcmp",
  },
  {
    Action: "-mldst-motion",
    Description: "MergedLoadStoreMotion",
  },
  {
    Action: "-sancov",
    Description: "Pass for instrumenting coverage on functions",
  },
  {
    Action: "-name-anon-globals",
    Description: "Provide a name to nameless globals",
  },
  {
    Action: "-nary-reassociate",
    Description: "Nary reassociation",
  },
  {
    Action: "-newgvn",
    Description: "Global Value Numbering",
  },
  {
    Action: "-pgo-memop-opt",
    Description: "Optimize memory intrinsic using its size value profile",
  },
  {
    Action: "-partial-inliner",
    Description: "Partial Inliner",
  },
  {
    Action: "-partially-inline-libcalls",
    Description: "Partially inline calls to library functions",
  },
  {
    Action: "-post-inline-ee-instrument",
    Description:
      "Instrument function entry/exit with calls to e.g. mcount()” “(post inlining)",
  },
  {
    Action: "-functionattrs",
    Description: "Deduce function attributes",
  },
  {
    Action: "-mem2reg",
    Description: "Promote Memory to ” “Register",
  },
  {
    Action: "-prune-eh",
    Description: "Remove unused exception handling info",
  },
  {
    Action: "-reassociate",
    Description: "Reassociate expressions",
  },
  {
    Action: "-redundant-dbg-inst-elim",
    Description: "Redundant Dbg Instruction Elimination",
  },
  {
    Action: "-rpo-functionattrs",
    Description: "Deduce function attributes in RPO",
  },
  {
    Action: "-rewrite-statepoints-for-gc",
    Description: "Make relocations explicit at statepoints",
  },
  {
    Action: "-sccp",
    Description: "Sparse Conditional Constant Propagation",
  },
  {
    Action: "-slp-vectorizer",
    Description: "SLP Vectorizer",
  },
  {
    Action: "-sroa",
    Description: "Scalar Replacement Of Aggregates",
  },
  {
    Action: "-scalarizer",
    Description: "Scalarize vector operations",
  },
  {
    Action: "-separate-const-offset-from-gep",
    Description:
      "Split GEPs to a variadic base and a constant offset for better CSE",
  },
  {
    Action: "-simple-loop-unswitch",
    Description: "Simple unswitch loops",
  },
  {
    Action: "-sink",
    Description: "Code sinking",
  },
  {
    Action: "-speculative-execution",
    Description: "Speculatively execute instructions",
  },
  {
    Action: "-slsr",
    Description: "Straight line strength reduction",
  },
  {
    Action: "-strip-dead-prototypes",
    Description: "Strip Unused Function Prototypes",
  },
  {
    Action: "-strip-debug-declare",
    Description: "Strip all llvm.dbg.declare intrinsics",
  },
  {
    Action: "-strip-nondebug",
    Description: "Strip all symbols, except dbg symbols, from a module",
  },
  {
    Action: "-strip",
    Description: "Strip all symbols from a module",
  },
  {
    Action: "-tailcallelim",
    Description: "Tail Call Elimination",
  },
  {
    Action: "-mergereturn",
    Description: "Unify function exit nodes",
  },
];
export default actionsDict;

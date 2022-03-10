// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the LICENSE file
// in the root directory of this source tree.
//
// This file was generated automatically the script
// build_tools/llvm/legacy_pass_manager/make_action_space_genfiles.py.

#define HANDLE_ACTION(action, handlePass)                              \
  switch (action) {                                                    \
    case LlvmAction::ADD_DISCRIMINATORS:                               \
      handlePass(llvm::createAddDiscriminatorsPass());                 \
      break;                                                           \
    case LlvmAction::ADCE:                                             \
      handlePass(llvm::createAggressiveDCEPass());                     \
      break;                                                           \
    case LlvmAction::AGGRESSIVE_INSTCOMBINE:                           \
      handlePass(llvm::createAggressiveInstCombinerPass());            \
      break;                                                           \
    case LlvmAction::ALIGNMENT_FROM_ASSUMPTIONS:                       \
      handlePass(llvm::createAlignmentFromAssumptionsPass());          \
      break;                                                           \
    case LlvmAction::ALWAYS_INLINE:                                    \
      handlePass(llvm::createAlwaysInlinerLegacyPass());               \
      break;                                                           \
    case LlvmAction::ARGPROMOTION:                                     \
      handlePass(llvm::createArgumentPromotionPass());                 \
      break;                                                           \
    case LlvmAction::ATTRIBUTOR:                                       \
      handlePass(llvm::createAttributorLegacyPass());                  \
      break;                                                           \
    case LlvmAction::BARRIER:                                          \
      handlePass(llvm::createBarrierNoopPass());                       \
      break;                                                           \
    case LlvmAction::BDCE:                                             \
      handlePass(llvm::createBitTrackingDCEPass());                    \
      break;                                                           \
    case LlvmAction::BREAK_CRIT_EDGES:                                 \
      handlePass(llvm::createBreakCriticalEdgesPass());                \
      break;                                                           \
    case LlvmAction::SIMPLIFYCFG:                                      \
      handlePass(llvm::createCFGSimplificationPass());                 \
      break;                                                           \
    case LlvmAction::CALLSITE_SPLITTING:                               \
      handlePass(llvm::createCallSiteSplittingPass());                 \
      break;                                                           \
    case LlvmAction::CALLED_VALUE_PROPAGATION:                         \
      handlePass(llvm::createCalledValuePropagationPass());            \
      break;                                                           \
    case LlvmAction::CANONICALIZE_ALIASES:                             \
      handlePass(llvm::createCanonicalizeAliasesPass());               \
      break;                                                           \
    case LlvmAction::CONSTHOIST:                                       \
      handlePass(llvm::createConstantHoistingPass());                  \
      break;                                                           \
    case LlvmAction::CONSTMERGE:                                       \
      handlePass(llvm::createConstantMergePass());                     \
      break;                                                           \
    case LlvmAction::CONSTPROP:                                        \
      handlePass(llvm::createConstantPropagationPass());               \
      break;                                                           \
    case LlvmAction::CORO_CLEANUP:                                     \
      handlePass(llvm::createCoroCleanupLegacyPass());                 \
      break;                                                           \
    case LlvmAction::CORO_EARLY:                                       \
      handlePass(llvm::createCoroEarlyLegacyPass());                   \
      break;                                                           \
    case LlvmAction::CORO_ELIDE:                                       \
      handlePass(llvm::createCoroElideLegacyPass());                   \
      break;                                                           \
    case LlvmAction::CORO_SPLIT:                                       \
      handlePass(llvm::createCoroSplitLegacyPass());                   \
      break;                                                           \
    case LlvmAction::CORRELATED_PROPAGATION:                           \
      handlePass(llvm::createCorrelatedValuePropagationPass());        \
      break;                                                           \
    case LlvmAction::CROSS_DSO_CFI:                                    \
      handlePass(llvm::createCrossDSOCFIPass());                       \
      break;                                                           \
    case LlvmAction::DEADARGELIM:                                      \
      handlePass(llvm::createDeadArgEliminationPass());                \
      break;                                                           \
    case LlvmAction::DCE:                                              \
      handlePass(llvm::createDeadCodeEliminationPass());               \
      break;                                                           \
    case LlvmAction::DIE:                                              \
      handlePass(llvm::createDeadInstEliminationPass());               \
      break;                                                           \
    case LlvmAction::DSE:                                              \
      handlePass(llvm::createDeadStoreEliminationPass());              \
      break;                                                           \
    case LlvmAction::REG2MEM:                                          \
      handlePass(llvm::createDemoteRegisterToMemoryPass());            \
      break;                                                           \
    case LlvmAction::DIV_REM_PAIRS:                                    \
      handlePass(llvm::createDivRemPairsPass());                       \
      break;                                                           \
    case LlvmAction::EARLY_CSE_MEMSSA:                                 \
      handlePass(llvm::createEarlyCSEMemSSAPass());                    \
      break;                                                           \
    case LlvmAction::EARLY_CSE:                                        \
      handlePass(llvm::createEarlyCSEPass());                          \
      break;                                                           \
    case LlvmAction::ELIM_AVAIL_EXTERN:                                \
      handlePass(llvm::createEliminateAvailableExternallyPass());      \
      break;                                                           \
    case LlvmAction::EE_INSTRUMENT:                                    \
      handlePass(llvm::createEntryExitInstrumenterPass());             \
      break;                                                           \
    case LlvmAction::FLATTENCFG:                                       \
      handlePass(llvm::createFlattenCFGPass());                        \
      break;                                                           \
    case LlvmAction::FLOAT2INT:                                        \
      handlePass(llvm::createFloat2IntPass());                         \
      break;                                                           \
    case LlvmAction::FORCEATTRS:                                       \
      handlePass(llvm::createForceFunctionAttrsLegacyPass());          \
      break;                                                           \
    case LlvmAction::INLINE:                                           \
      handlePass(llvm::createFunctionInliningPass());                  \
      break;                                                           \
    case LlvmAction::INSERT_GCOV_PROFILING:                            \
      handlePass(llvm::createGCOVProfilerPass());                      \
      break;                                                           \
    case LlvmAction::GVN_HOIST:                                        \
      handlePass(llvm::createGVNHoistPass());                          \
      break;                                                           \
    case LlvmAction::GVN:                                              \
      handlePass(llvm::createGVNPass());                               \
      break;                                                           \
    case LlvmAction::GLOBALDCE:                                        \
      handlePass(llvm::createGlobalDCEPass());                         \
      break;                                                           \
    case LlvmAction::GLOBALOPT:                                        \
      handlePass(llvm::createGlobalOptimizerPass());                   \
      break;                                                           \
    case LlvmAction::GLOBALSPLIT:                                      \
      handlePass(llvm::createGlobalSplitPass());                       \
      break;                                                           \
    case LlvmAction::GUARD_WIDENING:                                   \
      handlePass(llvm::createGuardWideningPass());                     \
      break;                                                           \
    case LlvmAction::HOTCOLDSPLIT:                                     \
      handlePass(llvm::createHotColdSplittingPass());                  \
      break;                                                           \
    case LlvmAction::IPCONSTPROP:                                      \
      handlePass(llvm::createIPConstantPropagationPass());             \
      break;                                                           \
    case LlvmAction::IPSCCP:                                           \
      handlePass(llvm::createIPSCCPPass());                            \
      break;                                                           \
    case LlvmAction::INDVARS:                                          \
      handlePass(llvm::createIndVarSimplifyPass());                    \
      break;                                                           \
    case LlvmAction::IRCE:                                             \
      handlePass(llvm::createInductiveRangeCheckEliminationPass());    \
      break;                                                           \
    case LlvmAction::INFER_ADDRESS_SPACES:                             \
      handlePass(llvm::createInferAddressSpacesPass());                \
      break;                                                           \
    case LlvmAction::INFERATTRS:                                       \
      handlePass(llvm::createInferFunctionAttrsLegacyPass());          \
      break;                                                           \
    case LlvmAction::INJECT_TLI_MAPPINGS:                              \
      handlePass(llvm::createInjectTLIMappingsLegacyPass());           \
      break;                                                           \
    case LlvmAction::INSTSIMPLIFY:                                     \
      handlePass(llvm::createInstSimplifyLegacyPass());                \
      break;                                                           \
    case LlvmAction::INSTCOMBINE:                                      \
      handlePass(llvm::createInstructionCombiningPass());              \
      break;                                                           \
    case LlvmAction::INSTNAMER:                                        \
      handlePass(llvm::createInstructionNamerPass());                  \
      break;                                                           \
    case LlvmAction::JUMP_THREADING:                                   \
      handlePass(llvm::createJumpThreadingPass());                     \
      break;                                                           \
    case LlvmAction::LCSSA:                                            \
      handlePass(llvm::createLCSSAPass());                             \
      break;                                                           \
    case LlvmAction::LICM:                                             \
      handlePass(llvm::createLICMPass());                              \
      break;                                                           \
    case LlvmAction::LIBCALLS_SHRINKWRAP:                              \
      handlePass(llvm::createLibCallsShrinkWrapPass());                \
      break;                                                           \
    case LlvmAction::LOAD_STORE_VECTORIZER:                            \
      handlePass(llvm::createLoadStoreVectorizerPass());               \
      break;                                                           \
    case LlvmAction::LOOP_DATA_PREFETCH:                               \
      handlePass(llvm::createLoopDataPrefetchPass());                  \
      break;                                                           \
    case LlvmAction::LOOP_DELETION:                                    \
      handlePass(llvm::createLoopDeletionPass());                      \
      break;                                                           \
    case LlvmAction::LOOP_DISTRIBUTE:                                  \
      handlePass(llvm::createLoopDistributePass());                    \
      break;                                                           \
    case LlvmAction::LOOP_FUSION:                                      \
      handlePass(llvm::createLoopFusePass());                          \
      break;                                                           \
    case LlvmAction::LOOP_GUARD_WIDENING:                              \
      handlePass(llvm::createLoopGuardWideningPass());                 \
      break;                                                           \
    case LlvmAction::LOOP_IDIOM:                                       \
      handlePass(llvm::createLoopIdiomPass());                         \
      break;                                                           \
    case LlvmAction::LOOP_INSTSIMPLIFY:                                \
      handlePass(llvm::createLoopInstSimplifyPass());                  \
      break;                                                           \
    case LlvmAction::LOOP_INTERCHANGE:                                 \
      handlePass(llvm::createLoopInterchangePass());                   \
      break;                                                           \
    case LlvmAction::LOOP_LOAD_ELIM:                                   \
      handlePass(llvm::createLoopLoadEliminationPass());               \
      break;                                                           \
    case LlvmAction::LOOP_PREDICATION:                                 \
      handlePass(llvm::createLoopPredicationPass());                   \
      break;                                                           \
    case LlvmAction::LOOP_REROLL:                                      \
      handlePass(llvm::createLoopRerollPass());                        \
      break;                                                           \
    case LlvmAction::LOOP_ROTATE:                                      \
      handlePass(llvm::createLoopRotatePass());                        \
      break;                                                           \
    case LlvmAction::LOOP_SIMPLIFYCFG:                                 \
      handlePass(llvm::createLoopSimplifyCFGPass());                   \
      break;                                                           \
    case LlvmAction::LOOP_SIMPLIFY:                                    \
      handlePass(llvm::createLoopSimplifyPass());                      \
      break;                                                           \
    case LlvmAction::LOOP_SINK:                                        \
      handlePass(llvm::createLoopSinkPass());                          \
      break;                                                           \
    case LlvmAction::LOOP_REDUCE:                                      \
      handlePass(llvm::createLoopStrengthReducePass());                \
      break;                                                           \
    case LlvmAction::LOOP_UNROLL_AND_JAM:                              \
      handlePass(llvm::createLoopUnrollAndJamPass());                  \
      break;                                                           \
    case LlvmAction::LOOP_UNROLL:                                      \
      handlePass(llvm::createLoopUnrollPass());                        \
      break;                                                           \
    case LlvmAction::LOOP_UNSWITCH:                                    \
      handlePass(llvm::createLoopUnswitchPass());                      \
      break;                                                           \
    case LlvmAction::LOOP_VECTORIZE:                                   \
      handlePass(llvm::createLoopVectorizePass());                     \
      break;                                                           \
    case LlvmAction::LOOP_VERSIONING_LICM:                             \
      handlePass(llvm::createLoopVersioningLICMPass());                \
      break;                                                           \
    case LlvmAction::LOOP_VERSIONING:                                  \
      handlePass(llvm::createLoopVersioningPass());                    \
      break;                                                           \
    case LlvmAction::LOWERATOMIC:                                      \
      handlePass(llvm::createLowerAtomicPass());                       \
      break;                                                           \
    case LlvmAction::LOWER_CONSTANT_INTRINSICS:                        \
      handlePass(llvm::createLowerConstantIntrinsicsPass());           \
      break;                                                           \
    case LlvmAction::LOWER_EXPECT:                                     \
      handlePass(llvm::createLowerExpectIntrinsicPass());              \
      break;                                                           \
    case LlvmAction::LOWER_GUARD_INTRINSIC:                            \
      handlePass(llvm::createLowerGuardIntrinsicPass());               \
      break;                                                           \
    case LlvmAction::LOWERINVOKE:                                      \
      handlePass(llvm::createLowerInvokePass());                       \
      break;                                                           \
    case LlvmAction::LOWER_MATRIX_INTRINSICS:                          \
      handlePass(llvm::createLowerMatrixIntrinsicsPass());             \
      break;                                                           \
    case LlvmAction::LOWERSWITCH:                                      \
      handlePass(llvm::createLowerSwitchPass());                       \
      break;                                                           \
    case LlvmAction::LOWER_WIDENABLE_CONDITION:                        \
      handlePass(llvm::createLowerWidenableConditionPass());           \
      break;                                                           \
    case LlvmAction::MEMCPYOPT:                                        \
      handlePass(llvm::createMemCpyOptPass());                         \
      break;                                                           \
    case LlvmAction::MERGEFUNC:                                        \
      handlePass(llvm::createMergeFunctionsPass());                    \
      break;                                                           \
    case LlvmAction::MERGEICMPS:                                       \
      handlePass(llvm::createMergeICmpsLegacyPass());                  \
      break;                                                           \
    case LlvmAction::MLDST_MOTION:                                     \
      handlePass(llvm::createMergedLoadStoreMotionPass());             \
      break;                                                           \
    case LlvmAction::SANCOV:                                           \
      handlePass(llvm::createModuleSanitizerCoverageLegacyPassPass()); \
      break;                                                           \
    case LlvmAction::NAME_ANON_GLOBALS:                                \
      handlePass(llvm::createNameAnonGlobalPass());                    \
      break;                                                           \
    case LlvmAction::NARY_REASSOCIATE:                                 \
      handlePass(llvm::createNaryReassociatePass());                   \
      break;                                                           \
    case LlvmAction::NEWGVN:                                           \
      handlePass(llvm::createNewGVNPass());                            \
      break;                                                           \
    case LlvmAction::PGO_MEMOP_OPT:                                    \
      handlePass(llvm::createPGOMemOPSizeOptLegacyPass());             \
      break;                                                           \
    case LlvmAction::PARTIAL_INLINER:                                  \
      handlePass(llvm::createPartialInliningPass());                   \
      break;                                                           \
    case LlvmAction::PARTIALLY_INLINE_LIBCALLS:                        \
      handlePass(llvm::createPartiallyInlineLibCallsPass());           \
      break;                                                           \
    case LlvmAction::POST_INLINE_EE_INSTRUMENT:                        \
      handlePass(llvm::createPostInlineEntryExitInstrumenterPass());   \
      break;                                                           \
    case LlvmAction::FUNCTIONATTRS:                                    \
      handlePass(llvm::createPostOrderFunctionAttrsLegacyPass());      \
      break;                                                           \
    case LlvmAction::MEM2REG:                                          \
      handlePass(llvm::createPromoteMemoryToRegisterPass());           \
      break;                                                           \
    case LlvmAction::PRUNE_EH:                                         \
      handlePass(llvm::createPruneEHPass());                           \
      break;                                                           \
    case LlvmAction::REASSOCIATE:                                      \
      handlePass(llvm::createReassociatePass());                       \
      break;                                                           \
    case LlvmAction::REDUNDANT_DBG_INST_ELIM:                          \
      handlePass(llvm::createRedundantDbgInstEliminationPass());       \
      break;                                                           \
    case LlvmAction::RPO_FUNCTIONATTRS:                                \
      handlePass(llvm::createReversePostOrderFunctionAttrsPass());     \
      break;                                                           \
    case LlvmAction::REWRITE_STATEPOINTS_FOR_GC:                       \
      handlePass(llvm::createRewriteStatepointsForGCLegacyPass());     \
      break;                                                           \
    case LlvmAction::SCCP:                                             \
      handlePass(llvm::createSCCPPass());                              \
      break;                                                           \
    case LlvmAction::SLP_VECTORIZER:                                   \
      handlePass(llvm::createSLPVectorizerPass());                     \
      break;                                                           \
    case LlvmAction::SROA:                                             \
      handlePass(llvm::createSROAPass());                              \
      break;                                                           \
    case LlvmAction::SCALARIZER:                                       \
      handlePass(llvm::createScalarizerPass());                        \
      break;                                                           \
    case LlvmAction::SEPARATE_CONST_OFFSET_FROM_GEP:                   \
      handlePass(llvm::createSeparateConstOffsetFromGEPPass());        \
      break;                                                           \
    case LlvmAction::SIMPLE_LOOP_UNSWITCH:                             \
      handlePass(llvm::createSimpleLoopUnswitchLegacyPass());          \
      break;                                                           \
    case LlvmAction::SINK:                                             \
      handlePass(llvm::createSinkingPass());                           \
      break;                                                           \
    case LlvmAction::SPECULATIVE_EXECUTION:                            \
      handlePass(llvm::createSpeculativeExecutionPass());              \
      break;                                                           \
    case LlvmAction::SLSR:                                             \
      handlePass(llvm::createStraightLineStrengthReducePass());        \
      break;                                                           \
    case LlvmAction::STRIP_DEAD_PROTOTYPES:                            \
      handlePass(llvm::createStripDeadPrototypesPass());               \
      break;                                                           \
    case LlvmAction::STRIP_DEBUG_DECLARE:                              \
      handlePass(llvm::createStripDebugDeclarePass());                 \
      break;                                                           \
    case LlvmAction::STRIP_NONDEBUG:                                   \
      handlePass(llvm::createStripNonDebugSymbolsPass());              \
      break;                                                           \
    case LlvmAction::STRIP:                                            \
      handlePass(llvm::createStripSymbolsPass());                      \
      break;                                                           \
    case LlvmAction::TAILCALLELIM:                                     \
      handlePass(llvm::createTailCallEliminationPass());               \
      break;                                                           \
    case LlvmAction::MERGERETURN:                                      \
      handlePass(llvm::createUnifyFunctionExitNodesPass());            \
      break;                                                           \
  }

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Configuration for building an action space from a list of LLVM passes."""
from compiler_gym.envs.llvm.service.passes.common import Pass

# A set of headers that must be included.
EXTRA_LLVM_HEADERS = {
    "include/llvm/LinkAllPasses.h",
    "include/llvm/Transforms/Coroutines.h",
    "include/llvm/Transforms/IPO.h",
    "include/llvm/Transforms/Scalar.h",
    "include/llvm/Transforms/Utils.h",
}

# A mapping from the name of a pass as defined in a INITIALIZE_PASS(name, ...)
# macro invocation to the name of the pass as defined in the createPASS();
# factory function. Not all passes are named consistently.
CREATE_PASS_NAME_MAP = {
    "ADCELegacyPass": "AggressiveDCEPass",
    "AddDiscriminatorsLegacyPass": "AddDiscriminatorsPass",
    "AggressiveInstCombinerLegacyPass": "AggressiveInstCombinerPass",
    "AlignmentFromAssumptions": "AlignmentFromAssumptionsPass",
    "ArgPromotion": "ArgumentPromotionPass",
    "BarrierNoop": "BarrierNoopPass",
    "BDCELegacyPass": "BitTrackingDCEPass",
    "BlockExtractor": "BlockExtractorPass",
    "BreakCriticalEdges": "BreakCriticalEdgesPass",
    "CalledValuePropagationLegacyPass": "CalledValuePropagationPass",
    "CallSiteSplittingLegacyPass": "CallSiteSplittingPass",
    "CanonicalizeAliasesLegacyPass": "CanonicalizeAliasesPass",
    "CFGSimplifyPass": "CFGSimplificationPass",
    "CFGuard": ["CFGuardCheckPass", "CFGuardDispatchPass"],
    "ConstantHoistingLegacyPass": "ConstantHoistingPass",
    "ConstantMergeLegacyPass": "ConstantMergePass",
    "ConstantPropagation": "ConstantPropagationPass",
    "CoroCleanupLegacy": "CoroCleanupLegacyPass",
    "CoroEarlyLegacy": "CoroEarlyLegacyPass",
    "CoroElideLegacy": "CoroElideLegacyPass",
    "CoroSplitLegacy": "CoroSplitLegacyPass",
    "CorrelatedValuePropagation": "CorrelatedValuePropagationPass",
    "CrossDSOCFI": "CrossDSOCFIPass",
    "DAE": "DeadArgEliminationPass",
    "DataFlowSanitizer": "DataFlowSanitizerPass",
    "DCELegacyPass": "DeadCodeEliminationPass",
    "DeadInstElimination": "DeadInstEliminationPass",
    "DivRemPairsLegacyPass": "DivRemPairsPass",
    "DSELegacyPass": "DeadStoreEliminationPass",
    "EarlyCSEMemSSALegacyPass": "EarlyCSEPass",
    "EliminateAvailableExternallyLegacyPass": "EliminateAvailableExternallyPass",
    "EntryExitInstrumenter": "EntryExitInstrumenterPass",
    "Float2IntLegacyPass": "Float2IntPass",
    "FunctionImportLegacyPass": "FunctionImportPass",
    "GCOVProfilerLegacyPass": "GCOVProfilerPass",
    "GlobalDCELegacyPass": "GlobalDCEPass",
    "GlobalOptLegacyPass": "GlobalOptimizerPass",
    "GlobalSplit": "GlobalSplitPass",
    "GuardWideningLegacyPass": "GuardWideningPass",
    "GVNHoistLegacyPass": "GVNHoistPass",
    "GVNLegacyPass": "GVNPass",
    "GVNSinkLegacyPass": "GVNSinkPass",
    "HotColdSplittingLegacyPass": "HotColdSplittingPass",
    "ICPPass": "IPConstantPropagationPass",
    "IndVarSimplifyLegacyPass": "IndVarSimplifyPass",
    "InferAddressSpaces": "InferAddressSpacesPass",
    "InjectTLIMappingsLegacy": "InjectTLIMappingsLegacyPass",
    "InstNamer": "InstructionNamerPass",
    "InstrOrderFileLegacyPass": "InstrOrderFilePass",
    "InternalizeLegacyPass": "InternalizePass",
    "IPCP": "IPConstantPropagationPass",
    "IPSCCPLegacyPass": "IPSCCPPass",
    "IRCELegacyPass": "InductiveRangeCheckEliminationPass",
    "JumpThreading": "JumpThreadingPass",
    "LCSSAWrapperPass": "LCSSAPass",
    "LegacyLICMPass": "LICMPass",
    "LegacyLoopSinkPass": "LoopSinkPass",
    "LibCallsShrinkWrapLegacyPass": "LibCallsShrinkWrapPass",
    "LoadStoreVectorizerLegacyPass": "LoadStoreVectorizerPass",
    "LoopDataPrefetchLegacyPass": "LoopDataPrefetchPass",
    "LoopDeletionLegacyPass": "LoopDeletionPass",
    "LoopDistributeLegacy": "LoopDistributePass",
    "LoopExtractor": "LoopExtractorPass",
    "LoopFuseLegacy": "LoopFusePass",
    "LoopGuardWideningLegacyPass": "LoopGuardWideningPass",
    "LoopIdiomRecognizeLegacyPass": "LoopIdiomPass",
    "LoopInstSimplifyLegacyPass": "LoopInstSimplifyPass",
    "LoopInterchange": "LoopInterchangePass",
    "LoopLoadElimination": "LoopLoadEliminationPass",
    "LoopPredicationLegacyPass": "LoopPredicationPass",
    "LoopReroll": "LoopRerollPass",
    "LoopRotateLegacyPass": "LoopRotatePass",
    "LoopSimplify": "LoopSimplifyPass",
    "LoopSimplifyCFGLegacyPass": "LoopSimplifyCFGPass",
    "LoopStrengthReduce": "LoopStrengthReducePass",
    "LoopUnroll": "LoopUnrollPass",
    "LoopUnrollAndJam": "LoopUnrollAndJamPass",
    "LoopUnswitch": "LoopUnswitchPass",
    "LoopVectorize": "LoopVectorizePass",
    "LoopVersioningLICM": "LoopVersioningLICMPass",
    "LowerAtomicLegacyPass": "LowerAtomicPass",
    "LowerConstantIntrinsics": "LowerConstantIntrinsicsPass",
    "LowerExpectIntrinsic": "LowerExpectIntrinsicPass",
    "LowerGuardIntrinsicLegacyPass": "LowerGuardIntrinsicPass",
    "LowerInvokeLegacyPass": "LowerInvokePass",
    "LowerMatrixIntrinsicsLegacyPass": "LowerMatrixIntrinsicsPass",
    "LowerSwitch": "LowerSwitchPass",
    "LowerWidenableConditionLegacyPass": "LowerWidenableConditionPass",
    "MemCpyOptLegacyPass": "MemCpyOptPass",
    "MemorySanitizerLegacyPass": "MemorySanitizerLegacyPassPass",
    "MergedLoadStoreMotionLegacyPass": "MergedLoadStoreMotionPass",
    "MergeFunctionsLegacyPass": "MergeFunctionsPass",
    "MetaRenamer": "MetaRenamerPass",
    "ModuleAddressSanitizerLegacyPass": "ModuleAddressSanitizerLegacyPassPass",
    "ModuleSanitizerCoverageLegacyPass": "ModuleSanitizerCoverageLegacyPassPass",
    "NameAnonGlobalLegacyPass": "NameAnonGlobalPass",
    "NaryReassociateLegacyPass": "NaryReassociatePass",
    "NewGVNLegacyPass": "NewGVNPass",
    "ObjCARCAPElim": "ObjCARCAPElimPass",
    "ObjCARCContract": "ObjCARCContractPass",
    "ObjCARCExpand": "ObjCARCExpandPass",
    "ObjCARCOpt": "ObjCARCOptPass",
    "PAEval": "PAEvalPass",
    "PartialInlinerLegacyPass": "PartialInliningPass",
    "PartiallyInlineLibCallsLegacyPass": "PartiallyInlineLibCallsPass",
    "PlaceSafepoints": "PlaceSafepointsPass",
    "PostInlineEntryExitInstrumenter": "PostInlineEntryExitInstrumenterPass",
    "PromoteLegacyPass": "PromoteMemoryToRegisterPass",
    "PruneEH": "PruneEHPass",
    "ReassociateLegacyPass": "ReassociatePass",
    "RedundantDbgInstElimination": "RedundantDbgInstEliminationPass",
    "RegToMem": "DemoteRegisterToMemoryPass",
    "ReversePostOrderFunctionAttrsLegacyPass": "ReversePostOrderFunctionAttrsPass",
    "RewriteSymbolsLegacyPass": "RewriteSymbolsPass",
    "SampleProfileLoaderLegacyPass": "SampleProfileLoaderPass",
    "ScalarizerLegacyPass": "ScalarizerPass",
    "SCCPLegacyPass": "SCCPPass",
    "SeparateConstOffsetFromGEP": "SeparateConstOffsetFromGEPPass",
    "SimpleInliner": "FunctionInliningPass",
    "SingleLoopExtractor": "SingleLoopExtractorPass",
    "SinkingLegacyPass": "SinkingPass",
    "SLPVectorizer": "SLPVectorizerPass",
    "SpeculativeExecutionLegacyPass": "SpeculativeExecutionPass",
    "SROALegacyPass": "SROAPass",
    "StraightLineStrengthReduce": "StraightLineStrengthReducePass",
    "StripDeadDebugInfo": "StripDeadDebugInfoPass",
    "StripDeadPrototypesLegacyPass": "StripDeadPrototypesPass",
    "StripDebugDeclare": "StripDebugDeclarePass",
    "StripNonDebugSymbols": "StripNonDebugSymbolsPass",
    "StripNonLineTableDebugInfo": "StripNonLineTableDebugInfoPass",
    "StripSymbols": "StripSymbolsPass",
    "StructurizeCFG": "StructurizeCFGPass",
    "TailCallElim": "TailCallEliminationPass",
    "ThreadSanitizerLegacyPass": "ThreadSanitizerLegacyPassPass",
    "UnifyFunctionExitNodes": "UnifyFunctionExitNodesPass",
}

# A list of pass names that should be excluded from the action space.
_EXCLUDED_PASSES = {
    # Irrelevant garbage collection passes.
    "StripGCRelocates",
    "PlaceBackedgeSafepointsImpl",
    "PlaceSafepointsPass",
    "RewriteStatepointsForGclegacyPass",
    # Irrelevant Objective-C Automatic Reference Counting passes.
    "ObjCARCAAWrapperPass",
    "ObjCARCAPElim",
    "ObjCARCAPElimPass",
    "ObjCARCContractPass",
    "ObjCARCExpandPass",
    "ObjCARCOptPass",
    # Doesn't use legacy pass constructor API, or requires additional
    # constructor arguments that are not available.
    "WholeProgramDevirt",
    "MakeGuardsExplicitLegacyPass",
    "LowerTypeTests",
    "EarlyCSELegacyPass",
    # Unneeded debugging passes.
    "WriteThinLTOBitcode",
    "PredicateInfoPrinterLegacyPass",
    "WarnMissedTransformationsLegacy",
    "DAH",  # Bugpoint only.
    "MetaRenamerPass",
    "PAEvalPass",
    "BarrierNoop",  # Used for debugging pass manager.
    "StripNonLineTableDebugInfoPass",  # Debug stripping.
    "StripDeadDebugInfoPass",  # Debug stripping.
    "LoopExtractorPass",  # Pulls out loops into functions. Changes semantics.
    "SingleLoopExtractorPass",  # Pulls out loops into functions. Changes semantics.
    "BlockExtractorPass",  # Pulls out blocks into functions. Changes semantics.
    # Unwanted instrumentation passes.
    "BoundsCheckingLegacyPass",  # Inserts traps on illegal access. Changes semantics.
    "ASanGlobalsMetadataWrapperPass",
    "AddressSanitizerLegacyPass",
    "HWAddressSanitizerLegacyPass",
    "SampleProfileLoaderPass",
    "MemorySanitizerLegacyPassPass",
    "ThreadSanitizerLegacyPassPass",
    "ModuleAddressSanitizerLegacyPassPass",
    "FunctionImportPass",
    "DataFlowSanitizerPass",
    "InstrOrderFilePass",
    "PostInlineEntryExitInstrumenter",
    # Profile-guided optimization or profiling.
    "PGOIndirectCallPromotionLegacyPass",
    "PGOInstrumentationUseLegacyPass",
    "PGOInstrumentationGenCreateVarLegacyPass",
    "PGOInstrumentationGenLegacyPass",
    "PGOInstrumentationUseLegacyPass",
    "PGOMemOpsizeOptLegacyPass",
    "PgomemOpsizeOptLegacyPass",
    "InstrProfilingLegacyPass",
    "ControlHeightReductionLegacyPass",
    # Unneeded symbol rewriting pass.
    "RewriteSymbolsPass",
    # Microsoft's Control Flow Guard checks on Windows targets.
    # https://llvm.org/doxygen/CFGuard_8cpp.html
    "CFGuardCheckPass",
    "CFGuardDispatchPass",
    # We don't want to change the visibility of symbols.
    "InternalizePass",
    # NOTE(github.com/facebookresearch/CompilerGym/issues/103): The
    # -structurizecg has been found to break the semantics of cBench benchmarks
    # ghostscript and tiff2bw.
    "StructurizeCFGPass",
    # NOTE(github.com/facebookresearch/CompilerGym/issues/46): The -gvn-sink
    # pass has been found to produce different states when run multiple times
    # on the same input.
    "GVNSinkPass",
}

# The name of the LLVM target to extract architecture-specific transforms for.
_TARGET = "X86"


def include_pass(pass_: Pass) -> bool:
    """Determine whether the pass should be included in the generated C++ sources."""
    if pass_.name in _EXCLUDED_PASSES:
        return False

    return "lib/Transforms" in pass_.source or f"Targets/{_TARGET}" in pass_.source

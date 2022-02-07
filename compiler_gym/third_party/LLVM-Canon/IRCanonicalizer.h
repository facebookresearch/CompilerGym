#pragma once

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"

/// \name Canonicalizer flags.
/// @{
/// Preserves original order of instructions.
extern llvm::cl::opt<bool> PreserveOrder;
/// Renames all instructions (including user-named).
extern llvm::cl::opt<bool> RenameAll;
/// Folds all regular instructions (including pre-outputs).
extern llvm::cl::opt<bool> FoldPreoutputs;
/// Sorts and reorders operands in commutative instructions.
extern llvm::cl::opt<bool> ReorderOperands;
/// @}

/// IRCanonicalizer aims to transform LLVM IR into canonical form.
class IRCanonicalizer : public llvm::FunctionPass {
 public:
  static char ID;

  /// Constructor for the IRCanonicalizer.
  ///
  /// \param PreserveOrder Preserves original order of instructions.
  /// \param RenameAll Renames all instructions (including user-named).
  /// \param FoldPreoutputs Folds all regular instructions (including pre-outputs).
  /// \param ReorderOperands Sorts and reorders operands in commutative instructions.
  IRCanonicalizer(bool PreserveOrder, bool RenameAll, bool FoldPreoutputs, bool ReorderOperands)
      : FunctionPass(ID),
        PreserveOrder(PreserveOrder),
        RenameAll(RenameAll),
        FoldPreoutputs(FoldPreoutputs),
        ReorderOperands(ReorderOperands) {}

  bool runOnFunction(llvm::Function& F) override;

 private:
  // Random constant for hashing, so the state isn't zero.
  const uint64_t MagicHashConstant = 0x6acaa36bef8325c5ULL;

  /// \name Canonicalizer flags.
  /// @{
  /// Preserves original order of instructions.
  bool PreserveOrder;
  /// Renames all instructions (including user-named).
  bool RenameAll;
  /// Folds all regular instructions (including pre-outputs).
  bool FoldPreoutputs;
  /// Sorts and reorders operands in commutative instructions.
  bool ReorderOperands;
  /// @}

  /// \name Naming.
  /// @{
  void nameFunctionArguments(llvm::Function& F);
  void nameBasicBlocks(llvm::Function& F);
  void nameInstructions(llvm::SmallVector<llvm::Instruction*, 16>& Outputs);
  void nameInstruction(llvm::Instruction* I,
                       llvm::SmallPtrSet<const llvm::Instruction*, 32>& Visited);
  void nameAsInitialInstruction(llvm::Instruction* I);
  void nameAsRegularInstruction(llvm::Instruction* I,
                                llvm::SmallPtrSet<const llvm::Instruction*, 32>& Visited);
  void foldInstructionName(llvm::Instruction* I);
  /// @}

  /// \name Reordering.
  /// @{
  void reorderInstructions(llvm::SmallVector<llvm::Instruction*, 16>& Outputs);
  void reorderInstruction(llvm::Instruction* Used, llvm::Instruction* User,
                          llvm::SmallPtrSet<const llvm::Instruction*, 32>& Visited);
  void reorderInstructionOperandsByNames(llvm::Instruction* I);
  void reorderPHIIncomingValues(llvm::PHINode* PN);
  /// @}

  /// \name Utility methods.
  /// @{
  llvm::SmallVector<llvm::Instruction*, 16> collectOutputInstructions(llvm::Function& F);
  bool isOutput(const llvm::Instruction* I);
  bool isInitialInstruction(const llvm::Instruction* I);
  bool hasOnlyImmediateOperands(const llvm::Instruction* I);
  llvm::SetVector<int> getOutputFootprint(llvm::Instruction* I,
                                          llvm::SmallPtrSet<const llvm::Instruction*, 32>& Visited);
  /// @}
};

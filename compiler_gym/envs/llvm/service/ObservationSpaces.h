// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <vector>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::llvm_service {

// The available observation spaces for LLVM.
//
// NOTE(cummins): Housekeeping rules - to add a new observation space:
//   1. Add a new entry to this LlvmObservationSpace enum.
//   2. Add a new switch case to getLlvmObservationSpaceList() to return the
//      ObserverationSpace.
//   3. Add a new switch case to LlvmSession::getObservation() to compute
//      the actual observation.
//   4. Run `bazel test //compiler_gym/...` and update the newly failing tests.
enum class LlvmObservationSpace {
  // The entire LLVM module as an IR string. This allows the user to do its own
  // feature extraction.
  IR,
  // Write the bitcode to a file. Returns a string, which is the path of the
  // written file.
  BITCODE_FILE,
  // The counts of all instructions in a program.
  INST_COUNT,
  // The Autophase feature vector from:
  //
  //   Huang, Q., Haj-Ali, A., Moses, W., Xiang, J., Stoica, I., Asanovic, K., &
  //   Wawrzynek, J. (2019). Autophase: Compiler phase-ordering for HLS with
  //   deep reinforcement learning. FCCM.
  AUTOPHASE,
  // Returns the graph representation of a program from:
  //
  //     Cummins, C., Fisches, Z. V., Ben-Nun, T., Hoefler, T., & Leather, H.
  //     (2020). ProGraML: Graph-based Deep Learning for Program Optimization
  //     and Analysis. ArXiv:2003.10536. https://arxiv.org/abs/2003.10536
  PROGRAML,
  // A JSON dictionary of properties describing the CPU.
  CPU_INFO,
  // The number of LLVM-IR instructions in the current module.
  IR_INSTRUCTION_COUNT,
  IR_INSTRUCTION_COUNT_O0,
  IR_INSTRUCTION_COUNT_O3,
  IR_INSTRUCTION_COUNT_OZ,
  // The size of the .text section of the lowered module. Platform dependent.
  OBJECT_TEXT_SIZE_BYTES,
  OBJECT_TEXT_SIZE_O0,
  OBJECT_TEXT_SIZE_O3,
  OBJECT_TEXT_SIZE_OZ,
#ifdef COMPILER_GYM_EXPERIMENTAL_TEXT_SIZE_COST
  // The size of the .text section of the compiled binary. Platform dependent.
  TEXT_SIZE_BYTES,
  TEXT_SIZE_O0,
  TEXT_SIZE_O3,
  TEXT_SIZE_OZ,
#endif
};

// Return the list of available observation spaces.
std::vector<ObservationSpace> getLlvmObservationSpaceList();

}  // namespace compiler_gym::llvm_service

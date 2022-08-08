// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <vector>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym::llvm_service {

/**
 * The available observation spaces for LLVM.
 *
 * \note Housekeeping rules - to add a new observation space:
 *   1. Add a new entry to this LlvmObservationSpace enum.
 *   2. Add a new switch case to getLlvmObservationSpaceList() to return the
 *      ObserverationSpace.
 *   3. Add a new switch case to LlvmSession::getObservation() to compute
 *      the actual observation.
 *   4. Run `bazel test //compiler_gym/...` and update the newly failing tests.
 */
enum class LlvmObservationSpace {
  /**
   * The entire LLVM module as an IR string.
   *
   * This allows the user to do their own feature extraction.
   */
  IR,
  /** The 40-digit hex SHA1 checksum of the LLVM module. */
  IR_SHA1,
  /** Get the bitcode as a bytes array. */
  BITCODE,
  /** Write the bitcode to a file and return its path as a string. */
  BITCODE_FILE,
  /** The counts of all instructions in a program. */
  INST_COUNT,
  /**
   * The Autophase feature vector.
   *
   * From:
   *
   *     Huang, Q., Haj-Ali, A., Moses, W., Xiang, J., Stoica, I., Asanovic, K.,
   *     & Wawrzynek, J. (2019). Autophase: Compiler phase-ordering for HLS with
   *     deep reinforcement learning. FCCM.
   */
  AUTOPHASE,
  /**
   * Returns the graph representation of a program as a networkx Graph.
   *
   * From:
   *
   *     Cummins, C., Fisches, Z. V., Ben-Nun, T., Hoefler, T., & Leather, H.
   *     (2020). ProGraML: Graph-based Deep Learning for Program Optimization
   *     and Analysis. ArXiv:2003.10536. https://arxiv.org/abs/2003.10536
   */
  PROGRAML,
  /**
   * Returns the graph representation of a program as a JSON node-link graph.
   *
   * From:
   *
   *     Cummins, C., Fisches, Z. V., Ben-Nun, T., Hoefler, T., & Leather, H.
   *     (2020). ProGraML: Graph-based Deep Learning for Program Optimization
   *     and Analysis. ArXiv:2003.10536. https://arxiv.org/abs/2003.10536
   */
  PROGRAML_JSON,
  /** A JSON dictionary of properties describing the CPU. */
  CPU_INFO,
  /** The number of LLVM-IR instructions in the current module. */
  IR_INSTRUCTION_COUNT,
  /** The number of LLVM-IR instructions normalized to `-O0`. */
  IR_INSTRUCTION_COUNT_O0,
  /** The number of LLVM-IR instructions normalized to `-O3`. */
  IR_INSTRUCTION_COUNT_O3,
  /** The number of LLVM-IR instructions normalized to `-Oz`. */
  IR_INSTRUCTION_COUNT_OZ,
  /** The platform-dependent size of the .text section of the lowered module. */
  OBJECT_TEXT_SIZE_BYTES,
  /** The platform-dependent size of the .text section of the lowered module. */
  OBJECT_TEXT_SIZE_O0,
  /** The platform-dependent size of the .text section of the lowered module. */
  OBJECT_TEXT_SIZE_O3,
  /** The platform-dependent size of the .text section of the lowered module. */
  OBJECT_TEXT_SIZE_OZ,
  /** The platform-dependent size of the .text section of the compiled binary. */
  TEXT_SIZE_BYTES,
  /** The platform-dependent size of the .text section of the compiled binary. */
  TEXT_SIZE_O0,
  /** The platform-dependent size of the .text section of the compiled binary. */
  TEXT_SIZE_O3,
  /** The platform-dependent size of the .text section of the compiled binary. */
  TEXT_SIZE_OZ,
  /** Return 1 if the benchmark is buildable, else 0.
   */
  IS_BUILDABLE,
  /** Return 1 if the benchmark is runnable, else 0.
   */
  IS_RUNNABLE,
  /** The runtime of the compiled program.
   *
   * Returns a list of runtime measurements in microseconds. This is not
   * available to all benchmarks. When not available, a list of zeros are returned.
   */
  RUNTIME,
  /** The time it took to compile the program.
   *
   * Returns a list of measurments in seconds. This is not available to all
   * benchmarks. When not available, a list of zeros are returned.
   */
  BUILDTIME,
  /** The LLVM-lexer token IDs of the input IR.
   *
   * Returns a dictionary of aligned lists (token_idx, token_kind,token_category, str_token_value)
   * one list element for every tokenized word in the IR.
   */
  LEXED_IR,
};

/** Return the list of available observation spaces. */
std::vector<ObservationSpace> getLlvmObservationSpaceList();

}  // namespace compiler_gym::llvm_service

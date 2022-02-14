// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include <glog/logging.h>

#include <iostream>

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"
#include "nlohmann/json.hpp"
#include "programl/graph/format/node_link_graph.h"
#include "programl/ir/llvm/llvm.h"

using nlohmann::json;

const programl::ProgramGraphOptions programlOptions;

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  CHECK(argc == 2) << "Usage: programl <bitcode-path>";

  auto buf = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (!buf) {
    LOG(FATAL) << "File not found: " << argv[1];
  }

  llvm::SMDiagnostic error;
  llvm::LLVMContext ctx;

  auto module = llvm::parseIRFile(argv[1], error, ctx);
  CHECK(module) << "Failed to parse: " << argv[1] << ": " << error.getMessage().str();

  // Build the ProGraML graph.
  programl::ProgramGraph graph;
  auto status = programl::ir::llvm::BuildProgramGraph(*module, &graph, programlOptions);
  if (!status.ok()) {
    return -1;
  }

  // Serialize the graph to a JSON node link graph.
  json nodeLinkGraph;
  status = programl::graph::format::ProgramGraphToNodeLinkGraph(graph, &nodeLinkGraph);
  if (!status.ok()) {
    return -1;
  }

  nodeLinkGraph.dump();

  return 0;
}

#include <glog/logging.h>

#include <iostream>

#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/SourceMgr.h"

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  CHECK(argc == 2) << "Usage: compute_autophase <bitcode-path>";

  auto buf = llvm::MemoryBuffer::getFileOrSTDIN(argv[1]);
  if (!buf) {
    LOG(FATAL) << "File not found: " << argv[1];
  }

  llvm::SMDiagnostic error;
  llvm::LLVMContext ctx;

  auto module = llvm::parseIRFile(argv[1], error, ctx);
  CHECK(module) << "Failed to parse: " << argv[1] << ": " << error.getMessage().str();

  std::cout << module->getInstructionCount() << std::endl;

  return 0;
}

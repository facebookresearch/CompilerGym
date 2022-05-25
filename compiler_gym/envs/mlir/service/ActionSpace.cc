// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "compiler_gym/envs/mlir/service/ActionSpace.h"

#include <fmt/format.h>
#include <stdint.h>

#include <boost/format.hpp>
#include <limits>
#include <magic_enum.hpp>

#include "compiler_gym/util/EnumUtil.h"
#include "compiler_gym/util/Unreachable.h"

namespace compiler_gym::mlir_service {

namespace {

Space getTileSizes() {
  Space res;
  Int64Box* box = res.mutable_int64_box();
  *box->mutable_low()->mutable_shape()->Add() = 3;
  *box->mutable_high()->mutable_shape()->Add() = 3;
  for (int i = 0; i < 3; ++i) {
    *box->mutable_low()->mutable_value()->Add() = 1;
    *box->mutable_high()->mutable_value()->Add() = std::numeric_limits<uint32_t>::max();
  }
  return res;
}

Space getInterchangeVector() {
  Space res;
  *res.mutable_type_id() = "permutation";
  res.mutable_int64_sequence()->mutable_length_range()->set_min(3);
  res.mutable_int64_sequence()->mutable_length_range()->set_max(3);
  res.mutable_int64_sequence()->mutable_scalar_range()->set_min(0);
  res.mutable_int64_sequence()->mutable_scalar_range()->set_max(2);
  return res;
}

Space getPromote() {
  Space res;
  res.mutable_boolean_value();
  return res;
}

Space getPromoteFullTile() {
  Space res;
  res.mutable_boolean_value();
  return res;
}

Space getLoopType() {
  Space res;
  NamedDiscreteSpace* space = res.mutable_named_discrete();
  *space->mutable_name()->Add() = "loops";
  *space->mutable_name()->Add() = "affine_loops";
  // Tailing fails with parallel loops. See
  // https://discourse.llvm.org/t/could-not-tile-namedstructuredop-in-linalg-whith-parallel-loop-type/60134
  //*space->mutable_name()->Add() = "parallel_loops";
  return res;
}

Space getTileOptionsSpace() {
  Space res;
  auto& map = *res.mutable_space_dict()->mutable_space();
  map["tile_sizes"] = getTileSizes();
  map["interchange_vector"] = getInterchangeVector();
  map["promote"] = getPromote();
  map["promote_full_tile"] = getPromoteFullTile();
  map["loop_type"] = getLoopType();
  return res;
}

Space getVectorizeTo() {
  Space res;
  NamedDiscreteSpace* space = res.mutable_named_discrete();
  // TODO(boian): query mmperf for these values.
  *space->mutable_name()->Add() = "dot";
  *space->mutable_name()->Add() = "matmul";
  *space->mutable_name()->Add() = "outer_product";
  return res;
}

Space getVectorTransferSplit() {
  Space res;
  NamedDiscreteSpace* space = res.mutable_named_discrete();
  // TODO(boian): query mmperf for these values.
  *space->mutable_name()->Add() = "none";
  *space->mutable_name()->Add() = "linalg_copy";
  *space->mutable_name()->Add() = "vector_transfer";
  return res;
}

Space getUnrollVectorTransfers() {
  Space res;
  res.mutable_boolean_value();
  return res;
}

Space getVectorizeOptions() {
  Space res;
  auto& map = *res.mutable_space_dict()->mutable_space();
  map["vectorize_to"] = getVectorizeTo();
  map["vector_transfer_split"] = getVectorTransferSplit();
  map["unroll_vector_transfers"] = getUnrollVectorTransfers();
  return res;
}

Space getMatMulOpSpace() {
  Space space;
  auto& map = *space.mutable_space_dict()->mutable_space();
  map["tile_options"] = getTileOptionsSpace();
  map["vectorize_options"] = getVectorizeOptions();
  return space;
}

Space getMatMulSpace() {
  Space space;
  space.mutable_space_sequence()->mutable_length_range()->set_min(1);
  space.mutable_space_sequence()->mutable_length_range()->set_max(4);
  *space.mutable_space_sequence()->mutable_space() = getMatMulOpSpace();
  return space;
}

ActionSpace getMatMulActionSpace() {
  ActionSpace space;
  space.set_name(
      util::enumNameToPascalCase<MlirActionSpace>(MlirActionSpace::MATRIX_MULTIPLICATION));
  *space.mutable_space() = getMatMulSpace();
  return space;
}

}  // namespace

std::vector<ActionSpace> getMlirActionSpaceList() {
  std::vector<ActionSpace> spaces;
  spaces.reserve(magic_enum::enum_count<MlirActionSpace>());

  for (const auto& value : magic_enum::enum_values<MlirActionSpace>()) {
    switch (value) {
      case MlirActionSpace::MATRIX_MULTIPLICATION: {
        spaces.push_back(getMatMulActionSpace());
      } break;
      default:
        UNREACHABLE(fmt::format("Unknown MLIR action space {}",
                                util::enumNameToPascalCase<MlirActionSpace>(value)));
    }
  }
  return spaces;
}

}  // namespace compiler_gym::mlir_service

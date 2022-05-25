// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "compiler_gym/service/proto/Proto.h"

#include <glog/logging.h>
#include <stdint.h>

#include <algorithm>
#include <cstddef>
#include <magic_enum.hpp>
#include <optional>

#define CG_PROTO_CHECK(cond, errorOnFalse) \
  do {                                     \
    if (errorOnFalse) {                    \
      CHECK(cond);                         \
    } else if (!(cond)) {                  \
      return false;                        \
    }                                      \
  } while (0)

#define CG_RETURN_IF_FALSE(cond) \
  do {                           \
    if (!(cond)) {               \
      return false;              \
    }                            \
  } while (0)

namespace compiler_gym {

namespace {

std::optional<std::string> getTypeId(const Space& space) {
  if (space.optional_type_id_case() == Space::OptionalTypeIdCase::kTypeId) {
    return space.type_id();
  } else {
    switch (space.value_case()) {
      case Space::ValueCase::kSpaceList:
        return std::string(magic_enum::enum_name(Space::ValueCase::kSpaceList));
      case Space::ValueCase::kSpaceDict:
        return std::string(magic_enum::enum_name(Space::ValueCase::kSpaceDict));
      case Space::ValueCase::kDiscrete:
        return std::string(magic_enum::enum_name(Space::ValueCase::kDiscrete));
      case Space::ValueCase::kNamedDiscrete:
        return std::string(magic_enum::enum_name(Space::ValueCase::kNamedDiscrete));
      case Space::ValueCase::kBooleanValue:
        return std::string(magic_enum::enum_name(Space::ValueCase::kBooleanValue));
      case Space::ValueCase::kInt64Value:
        return std::string(magic_enum::enum_name(Space::ValueCase::kInt64Value));
      case Space::ValueCase::kFloatValue:
        return std::string(magic_enum::enum_name(Space::ValueCase::kFloatValue));
      case Space::ValueCase::kDoubleValue:
        return std::string(magic_enum::enum_name(Space::ValueCase::kDoubleValue));
      case Space::ValueCase::kStringValue:
        return std::string(magic_enum::enum_name(Space::ValueCase::kStringValue));
      case Space::ValueCase::kBooleanSequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kBooleanSequence));
      case Space::ValueCase::kByteSequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kByteSequence));
      case Space::ValueCase::kBytesSequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kBytesSequence));
      case Space::ValueCase::kInt64Sequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kInt64Sequence));
      case Space::ValueCase::kFloatSequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kFloatSequence));
      case Space::ValueCase::kDoubleSequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kDoubleSequence));
      case Space::ValueCase::kStringSequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kStringSequence));
      case Space::ValueCase::kSpaceSequence:
        return std::string(magic_enum::enum_name(Space::ValueCase::kSpaceSequence));
      case Space::ValueCase::kBooleanBox:
        return std::string(magic_enum::enum_name(Space::ValueCase::kBooleanBox));
      case Space::ValueCase::kByteBox:
        return std::string(magic_enum::enum_name(Space::ValueCase::kByteBox));
      case Space::ValueCase::kInt64Box:
        return std::string(magic_enum::enum_name(Space::ValueCase::kInt64Box));
      case Space::ValueCase::kFloatBox:
        return std::string(magic_enum::enum_name(Space::ValueCase::kFloatBox));
      case Space::ValueCase::kDoubleBox:
        return std::string(magic_enum::enum_name(Space::ValueCase::kDoubleBox));
      case Space::ValueCase::kAnyValue:
        return std::string(magic_enum::enum_name(Space::ValueCase::kAnyValue));
      case Space::ValueCase::kSpaceUnion:
        return std::string(magic_enum::enum_name(Space::ValueCase::kSpaceUnion));
      default:
        return std::nullopt;
    }
  }
}

}  // anonymous namespace

bool spaceContains(const Space& space, const Event& event, bool errorOnFalse,
                   const SpaceContainsEventChecker::Context& ctx) {
  auto typeId = getTypeId(space);
  CG_PROTO_CHECK(bool(typeId), errorOnFalse);
  const auto& it = ctx.typeIdFuncMap.find(typeId.value());
  CHECK(it != ctx.typeIdFuncMap.end());
  return it->second(space, event, errorOnFalse, ctx);
}

bool spaceListContains(const Space& space, const Event& event, bool errorOnFalse,
                       const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kEventList, errorOnFalse);
  CG_PROTO_CHECK(event.event_list().event_size() == space.space_list().space_size(), errorOnFalse);
  for (size_t i = 0; i < event.event_list().event_size(); ++i) {
    CG_RETURN_IF_FALSE(
        spaceContains(space.space_list().space(i), event.event_list().event(i), errorOnFalse, ctx));
  }
  return true;
}

bool spaceDictContains(const Space& space, const Event& event, bool errorOnFalse,
                       const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kEventDict, errorOnFalse);
  CG_PROTO_CHECK(event.event_dict().event_size() == space.space_dict().space_size(), errorOnFalse);
  for (const auto& kv : event.event_dict().event()) {
    const auto& it = space.space_dict().space().find(kv.first);
    CG_PROTO_CHECK(it != space.space_dict().space().end(), errorOnFalse);
    CG_RETURN_IF_FALSE(spaceContains(it->second, kv.second, errorOnFalse, ctx));
  }
  return true;
}

bool discreteSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                           const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kInt64Value, errorOnFalse);
  CG_PROTO_CHECK(event.int64_value() < space.discrete().n(), errorOnFalse);
  return true;
}

bool namedDiscreteSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kInt64Value, errorOnFalse);
  CG_PROTO_CHECK(event.int64_value() < space.named_discrete().name_size(), errorOnFalse);
  return true;
}

bool booleanRangeContains(const Space& space, const Event& event, bool errorOnFalse,
                          const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kBooleanValue, errorOnFalse);
  CG_PROTO_CHECK(space.boolean_value().optional_min_case() ==
                         BooleanRange::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
                     space.boolean_value().min() <= event.boolean_value(),
                 errorOnFalse);
  CG_PROTO_CHECK(space.boolean_value().optional_max_case() ==
                         BooleanRange::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
                     event.boolean_value() <= space.boolean_value().max(),
                 errorOnFalse);
  return true;
}

bool int64RangeContains(const Space& space, const Event& event, bool errorOnFalse,
                        const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kInt64Value, errorOnFalse);
  CG_PROTO_CHECK(space.int64_value().optional_min_case() ==
                         Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
                     space.int64_value().min() <= event.int64_value(),
                 errorOnFalse);
  CG_PROTO_CHECK(space.int64_value().optional_max_case() ==
                         Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
                     event.int64_value() <= space.int64_value().max(),
                 errorOnFalse);
  return true;
}

bool floatRangeContains(const Space& space, const Event& event, bool errorOnFalse,
                        const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kFloatValue, errorOnFalse);
  CG_PROTO_CHECK(space.float_value().optional_min_case() ==
                         FloatRange::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
                     space.float_value().min() <= event.float_value(),
                 errorOnFalse);
  CG_PROTO_CHECK(space.float_value().optional_max_case() ==
                         FloatRange::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
                     event.float_value() <= space.float_value().max(),
                 errorOnFalse);
  return true;
}

bool doubleRangeContains(const Space& space, const Event& event, bool errorOnFalse,
                         const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kDoubleValue, errorOnFalse);
  CG_PROTO_CHECK(space.double_value().optional_min_case() ==
                         DoubleRange::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
                     space.double_value().min() <= event.double_value(),
                 errorOnFalse);
  CG_PROTO_CHECK(space.double_value().optional_max_case() ==
                         DoubleRange::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
                     event.double_value() <= space.double_value().max(),
                 errorOnFalse);
  return true;
}

bool stringSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                         const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kStringValue, errorOnFalse);
  CG_PROTO_CHECK(space.string_value().length_range().optional_min_case() ==
                         Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
                     space.string_value().length_range().min() <= event.string_value().size(),
                 errorOnFalse);
  CG_PROTO_CHECK(space.string_value().length_range().optional_max_case() ==
                         Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
                     event.string_value().size() <= space.string_value().length_range().max(),
                 errorOnFalse);
  return true;
}

bool booleanSequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                  const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kBooleanTensor, errorOnFalse);
  const auto& seq = space.boolean_sequence();
  const auto& tensor = event.boolean_tensor();
  CG_PROTO_CHECK(tensor.shape_size() == 1, errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_min_case() == Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
          seq.length_range().min() <= tensor.value_size(),
      errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_max_case() == Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
          tensor.value_size() <= seq.length_range().max(),
      errorOnFalse);
  for (const auto& v : tensor.value()) {
    CG_PROTO_CHECK(seq.scalar_range().optional_min_case() == BooleanRange::OPTIONAL_MIN_NOT_SET ||
                       seq.scalar_range().min() <= v,
                   errorOnFalse);
    CG_PROTO_CHECK(seq.scalar_range().optional_max_case() == BooleanRange::OPTIONAL_MAX_NOT_SET ||
                       v <= seq.scalar_range().max(),
                   errorOnFalse);
  }
  return true;
}

bool byteSequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                               const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kByteTensor, errorOnFalse);
  const auto& seq = space.byte_sequence();
  const auto& tensor = event.byte_tensor();
  CG_PROTO_CHECK(tensor.shape_size() == 1, errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_min_case() == Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
          seq.length_range().min() <= tensor.value().size(),
      errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_max_case() == Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
          tensor.value().size() <= seq.length_range().max(),
      errorOnFalse);
  for (const auto& v : tensor.value()) {
    CG_PROTO_CHECK(seq.scalar_range().optional_min_case() == Int64Range::OPTIONAL_MIN_NOT_SET ||
                       v <= seq.scalar_range().min(),
                   errorOnFalse);
    CG_PROTO_CHECK(seq.scalar_range().optional_max_case() == Int64Range::OPTIONAL_MAX_NOT_SET ||
                       v <= seq.scalar_range().max(),
                   errorOnFalse);
  }
  return true;
}

bool bytesSequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                const SpaceContainsEventChecker::Context& ctx) {
  // TODO(boian): Implement. Maybe use StringTensor or remove BytesSequenceSpace.
  CG_PROTO_CHECK(false && "Not implemented yet.", errorOnFalse);
  return true;
}

bool int64SequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kInt64Tensor, errorOnFalse);
  const auto& seq = space.int64_sequence();
  const auto& tensor = event.int64_tensor();
  CG_PROTO_CHECK(tensor.shape_size() == 1, errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_min_case() == Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
          seq.length_range().min() <= tensor.value_size(),
      errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_max_case() == Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
          tensor.value_size() <= seq.length_range().max(),
      errorOnFalse);
  for (const auto& v : tensor.value()) {
    CG_PROTO_CHECK(seq.scalar_range().optional_min_case() == Int64Range::OPTIONAL_MIN_NOT_SET ||
                       seq.scalar_range().min() <= v,
                   errorOnFalse);
    CG_PROTO_CHECK(seq.scalar_range().optional_max_case() == Int64Range::OPTIONAL_MAX_NOT_SET ||
                       v <= seq.scalar_range().max(),
                   errorOnFalse);
  }
  return true;
}

bool floatSequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kFloatTensor, errorOnFalse);
  const auto& seq = space.float_sequence();
  const auto& tensor = event.float_tensor();
  CG_PROTO_CHECK(tensor.shape_size() == 1, errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_min_case() == Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
          seq.length_range().min() <= tensor.value_size(),
      errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_max_case() == Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
          tensor.value_size() <= seq.length_range().max(),
      errorOnFalse);
  for (const auto& v : tensor.value()) {
    CG_PROTO_CHECK(seq.scalar_range().optional_min_case() == FloatRange::OPTIONAL_MIN_NOT_SET ||
                       seq.scalar_range().min() <= v,
                   errorOnFalse);
    CG_PROTO_CHECK(seq.scalar_range().optional_max_case() == FloatRange::OPTIONAL_MAX_NOT_SET ||
                       v <= seq.scalar_range().max(),
                   errorOnFalse);
  }
  return true;
}

bool doubleSequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                 const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kDoubleTensor, errorOnFalse);
  const auto& seq = space.double_sequence();
  const auto& tensor = event.double_tensor();
  CG_PROTO_CHECK(tensor.shape_size() == 1, errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_min_case() == Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
          seq.length_range().min() <= tensor.value_size(),
      errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_max_case() == Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
          tensor.value_size() <= seq.length_range().max(),
      errorOnFalse);
  for (const auto& v : tensor.value()) {
    CG_PROTO_CHECK(seq.scalar_range().optional_min_case() == DoubleRange::OPTIONAL_MIN_NOT_SET ||
                       seq.scalar_range().min() <= v,
                   errorOnFalse);
    CG_PROTO_CHECK(seq.scalar_range().optional_max_case() == DoubleRange::OPTIONAL_MAX_NOT_SET ||
                       v <= seq.scalar_range().max(),
                   errorOnFalse);
  }
  return true;
}

bool stringSequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                 const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kStringTensor, errorOnFalse);
  const auto& seq = space.string_sequence();
  const auto& tensor = event.string_tensor();
  CG_PROTO_CHECK(tensor.shape_size() == 1, errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_min_case() == Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
          seq.length_range().min() <= tensor.value_size(),
      errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_max_case() == Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
          tensor.value_size() <= seq.length_range().max(),
      errorOnFalse);
  return true;
}

bool spaceSequenceSpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                                const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kEventList, errorOnFalse);
  const auto& seq = space.space_sequence();
  const auto& list = event.event_list();
  CG_PROTO_CHECK(
      seq.length_range().optional_min_case() == Int64Range::OptionalMinCase::OPTIONAL_MIN_NOT_SET ||
          seq.length_range().min() <= list.event_size(),
      errorOnFalse);
  CG_PROTO_CHECK(
      seq.length_range().optional_max_case() == Int64Range::OptionalMaxCase::OPTIONAL_MAX_NOT_SET ||
          list.event_size() <= seq.length_range().max(),
      errorOnFalse);

  for (const auto& e : list.event()) {
    return spaceContains(seq.space(), e, errorOnFalse, ctx);
  }
  return true;
}

bool booleanBoxContains(const Space& space, const Event& event, bool errorOnFalse,
                        const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kBooleanTensor, errorOnFalse);
  const auto& box = space.boolean_box();
  const auto& tensor = event.boolean_tensor();
  CG_PROTO_CHECK(std::equal(box.low().shape().begin(), box.low().shape().end(),
                            tensor.shape().begin(), tensor.shape().end()),
                 errorOnFalse);
  for (size_t i = 0; i < tensor.value_size(); ++i) {
    CG_PROTO_CHECK(box.low().value(i) <= tensor.value(i) && tensor.value(i) <= box.high().value(i),
                   errorOnFalse);
  }
  return true;
}

bool byteBoxContains(const Space& space, const Event& event, bool errorOnFalse,
                     const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kByteTensor, errorOnFalse);
  const auto& box = space.byte_box();
  const auto& tensor = event.byte_tensor();
  CG_PROTO_CHECK(std::equal(box.low().shape().begin(), box.low().shape().end(),
                            tensor.shape().begin(), tensor.shape().end()),
                 errorOnFalse);
  for (size_t i = 0; i < tensor.value().size(); ++i) {
    CG_PROTO_CHECK(box.low().value().at(i) <= tensor.value().at(i) &&
                       tensor.value().at(i) <= box.high().value().at(i),
                   errorOnFalse);
  }
  return true;
}

bool int64BoxContains(const Space& space, const Event& event, bool errorOnFalse,
                      const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kInt64Tensor, errorOnFalse);
  const auto& box = space.int64_box();
  const auto& tensor = event.int64_tensor();
  CG_PROTO_CHECK(std::equal(box.low().shape().begin(), box.low().shape().end(),
                            tensor.shape().begin(), tensor.shape().end()),
                 errorOnFalse);
  for (size_t i = 0; i < tensor.value_size(); ++i) {
    CG_PROTO_CHECK(box.low().value(i) <= tensor.value(i) && tensor.value(i) <= box.high().value(i),
                   errorOnFalse);
  }
  return true;
}

bool floatBoxContains(const Space& space, const Event& event, bool errorOnFalse,
                      const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kFloatTensor, errorOnFalse);
  const auto& box = space.float_box();
  const auto& tensor = event.float_tensor();
  CG_PROTO_CHECK(std::equal(box.low().shape().begin(), box.low().shape().end(),
                            tensor.shape().begin(), tensor.shape().end()),
                 errorOnFalse);
  for (size_t i = 0; i < tensor.value_size(); ++i) {
    CG_PROTO_CHECK(box.low().value(i) <= tensor.value(i) && tensor.value(i) <= box.high().value(i),
                   errorOnFalse);
  }
  return true;
}

bool doubleBoxContains(const Space& space, const Event& event, bool errorOnFalse,
                       const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(event.value_case() == Event::kDoubleTensor, errorOnFalse);
  const auto& box = space.double_box();
  const auto& tensor = event.double_tensor();
  CG_PROTO_CHECK(std::equal(box.low().shape().begin(), box.low().shape().end(),
                            tensor.shape().begin(), tensor.shape().end()),
                 errorOnFalse);
  for (size_t i = 0; i < tensor.value_size(); ++i) {
    CG_PROTO_CHECK(box.low().value(i) <= tensor.value(i) && tensor.value(i) <= box.high().value(i),
                   errorOnFalse);
  }
  return true;
}

bool anySpaceContains(const Space& space, const Event& event, bool errorOnFalse,
                      const SpaceContainsEventChecker::Context& ctx) {
  // TODO(boian): implement
  CG_PROTO_CHECK(false, errorOnFalse);
}

bool spaceUnionContains(const Space& space, const Event& event, bool errorOnFalse,
                        const SpaceContainsEventChecker::Context& ctx) {
  for (const auto& s : space.space_union().space()) {
    if (spaceContains(s, event, false, ctx)) {
      return true;
    }
  }
  CG_PROTO_CHECK(false, errorOnFalse);
}

bool permutationContains(const Space& space, const Event& event, bool errorOnFalse,
                         const SpaceContainsEventChecker::Context& ctx) {
  CG_PROTO_CHECK(int64SequenceSpaceContains(space, event, errorOnFalse, ctx), errorOnFalse);
  std::vector<int64_t> permutation(event.int64_tensor().value().begin(),
                                   event.int64_tensor().value().end());
  std::sort(permutation.begin(), permutation.end());
  auto uniqueIt = std::unique(permutation.begin(), permutation.end());
  bool containsDuplicate = uniqueIt != permutation.end();
  CG_PROTO_CHECK(!containsDuplicate, errorOnFalse);
  CG_PROTO_CHECK(permutation.front() == 0, errorOnFalse);
  CG_PROTO_CHECK(permutation.back() == event.int64_tensor().value_size() - 1, errorOnFalse);
  return true;
}

bool SpaceContainsEventChecker::contains(const Space& space, const Event& event,
                                         bool errorOnFalse) const {
  return compiler_gym::spaceContains(space, event, errorOnFalse, ctx_);
}

void SpaceContainsEventChecker::checkContains(const Space& space, const Event& event) const {
  this->contains(space, event, true);
}

SpaceContainsEventChecker makeDefaultSpaceContainsEventChecker() {
  SpaceContainsEventChecker res;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kSpaceList))] =
      spaceListContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kSpaceDict))] =
      spaceDictContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kDiscrete))] =
      discreteSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kNamedDiscrete))] =
      namedDiscreteSpaceContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kBooleanValue))] =
      booleanRangeContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kInt64Value))] =
      int64RangeContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kFloatValue))] =
      floatRangeContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kDoubleValue))] =
      doubleRangeContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kStringValue))] =
      stringSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kBooleanSequence))] =
      booleanSequenceSpaceContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kByteSequence))] =
      byteSequenceSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kBytesSequence))] =
      bytesSequenceSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kInt64Sequence))] =
      int64SequenceSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kFloatSequence))] =
      floatSequenceSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kDoubleSequence))] =
      doubleSequenceSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kStringSequence))] =
      stringSequenceSpaceContains;
  res.context()
      .typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kSpaceSequence))] =
      spaceSequenceSpaceContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kBooleanBox))] =
      booleanBoxContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kByteBox))] =
      byteBoxContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kInt64Box))] =
      int64BoxContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kFloatBox))] =
      floatBoxContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kDoubleBox))] =
      doubleBoxContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kAnyValue))] =
      anySpaceContains;
  res.context().typeIdFuncMap[std::string(magic_enum::enum_name(Space::ValueCase::kSpaceUnion))] =
      spaceUnionContains;
  res.context().typeIdFuncMap["permutation"] = permutationContains;

  return res;
}

}  // namespace compiler_gym

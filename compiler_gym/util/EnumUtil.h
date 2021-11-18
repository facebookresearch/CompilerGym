// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <fmt/format.h>
#include <grpcpp/grpcpp.h>

#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "magic_enum.hpp"

namespace compiler_gym::util {

/**
 * Convert an UPPER_SNAKE_CASE enum name to PascalCase.
 *
 * E.g. `MyEnum::MY_ENUM_VALUE -> "MyEnumValue"`.
 *
 * @param value An enum.
 * @return A string.
 */
template <typename Enum>
std::string enumNameToPascalCase(Enum value) {
  const std::string name(magic_enum::enum_name<Enum>(value));
  std::string out;
  bool capitalize = true;
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '_') {
      capitalize = true;
    } else {
      out.push_back(capitalize ? toupper(name[i]) : tolower(name[i]));
      capitalize = false;
    }
  }
  return out;
}

/**
 * Convert an UPPER_SNAKE_CASE enum name to PascalCase.
 *
 * E.g. `MyEnum::MY_ENUM_VALUE -> "MyEnumValue"`.
 *
 * @param value An enum.
 * @return A string.
 */
template <typename Enum>
std::string enumNameToPascalCase(std::optional<Enum> value) {
  if (!value.has_value()) {
    return "None";
  }
  return enumNameToPascalCase(value.value());
}

/**
 * Convert an UPPER_SNAKE_CASE enum name to -flag-name.
 *
 * E.g. `MyEnum::MY_ENUM_VALUE -> "-my-enum-value"`.
 *
 * @param value An enum.
 * @return A string.
 */
template <typename Enum>
std::string enumNameToCommandlineFlag(Enum value) {
  const std::string name(magic_enum::enum_name<Enum>(value));
  std::string out{"-"};
  for (size_t i = 0; i < name.size(); ++i) {
    if (name[i] == '_') {
      out.push_back('-');
    } else {
      out.push_back(tolower(name[i]));
    }
  }
  return out;
}

/**
 * Enumerate all values of an optional Enum, including `std::nullopt`.
 *
 * @return A vector of optional enum values.
 */
template <typename Enum>
std::vector<std::optional<Enum>> optionalEnumValues() {
  std::vector<std::optional<Enum>> values;
  values.push_back(std::nullopt);
  for (const auto value : magic_enum::enum_values<Enum>()) {
    values.push_back(value);
  }
  return values;
}

/**
 * Return the name of an enum, e.g. `demangle<foo::MyEnum>() -> "MyEnum"`.
 *
 * @return A string.
 */
template <typename Enum>
std::string demangle() {
  const std::string name(magic_enum::enum_type_name<Enum>());
  const auto pos = name.rfind("::");
  if (pos == std::string::npos) {
    return name;
  } else {
    return name.substr(pos + 2);
  }
}

/**
 * Convert a PascalCase enum name to enum value.
 *
 * E.g. `pascalCaseToEnum("MyEnumVal", &myEnum) -> MyEnum::MY_ENUM_VAL`.
 *
 * @tparam Enum Enum type.
 * @param name A string.
 * @param value The value to write to.
 * @return `Status::OK` on success. `Status::INVALID_ARGUMENT` if the string
 *          name is not recognized.
 */
template <typename Enum>
[[nodiscard]] grpc::Status pascalCaseToEnum(const std::string& name, Enum* value) {
  for (const auto candidateValue : magic_enum::enum_values<Enum>()) {
    const std::string pascalCaseName = enumNameToPascalCase(candidateValue);
    if (pascalCaseName == name) {
      *value = candidateValue;
      return grpc::Status::OK;
    }
  }
  return grpc::Status(
      grpc::StatusCode::INVALID_ARGUMENT,
      fmt::format("Could not convert '{}' to {} enum entry", name, demangle<Enum>()));
}

/**
 * Create a map from PascalCase enum value names to enum values.
 *
 * @tparam Enum Enum type.
 * @return A `name -> value` lookup table.
 */
template <typename Enum>
std::unordered_map<std::string, Enum> createPascalCaseToEnumLookupTable() {
  std::unordered_map<std::string, Enum> table;
  for (const auto value : magic_enum::enum_values<Enum>()) {
    const std::string pascalCaseName = enumNameToPascalCase(value);
    table[pascalCaseName] = value;
  }
  return table;
}

/**
 * Convert an integer to an enum with bounds checking.
 *
 * E.g. `intToEnum(3, &myEnum);`
 *
 * @tparam Enum Enum type.
 * @param numericValue An integer.
 * @param enumValue An enum to write.
 * @return `Status::OK` on success. `Status::INVALID_ARGUMENT` if out of bounds.
 */
template <typename Enum>
[[nodiscard]] inline grpc::Status intToEnum(int numericValue, Enum* enumValue) {
  const auto max = magic_enum::enum_count<Enum>();
  if (numericValue < 0 || static_cast<decltype(max)>(numericValue) >= max) {
    return grpc::Status(
        grpc::StatusCode::INVALID_ARGUMENT,
        fmt::format("{}({}) is not in range [0, {})", demangle<Enum>(), numericValue, max));
  }
  *enumValue = static_cast<Enum>(numericValue);
  return grpc::Status::OK;
}

}  // namespace compiler_gym::util

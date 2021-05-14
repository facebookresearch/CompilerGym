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

// Convert an UPPER_SNAKE_CASE enum name to PascalCase.
// E.g. MyEnum::MY_ENUM_VALUE -> "MyEnumValue".
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

// Convert an optional UPPER_SNAKE_CASE enum name to PascalCase.
// E.g. MyEnum::MY_ENUM_VALUE -> "MyEnumValue".
template <typename Enum>
std::string enumNameToPascalCase(std::optional<Enum> value) {
  if (!value.has_value()) {
    return "None";
  }
  return enumNameToPascalCase(value.value());
}

// Enumerate all values of an optional enum, including nullopt.
template <typename Enum>
std::vector<std::optional<Enum>> optionalEnumValues() {
  std::vector<std::optional<Enum>> values;
  values.push_back(std::nullopt);
  for (const auto value : magic_enum::enum_values<Enum>()) {
    values.push_back(value);
  }
  return values;
}

// Return the name of an enum, e.g. demangle<foo::MyEnum>() -> "MyEnum".
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

// Convert a PascalCase enum name to enum value.
// E.g. pascalCaseToEnum("MyEnumVal", &myEnum) -> MyEnum::MY_ENUM_VAL
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

// Create a map from PascalCase enum value names to enum values.
template <typename Enum>
std::unordered_map<std::string, Enum> createPascalCaseToEnumLookupTable() {
  std::unordered_map<std::string, Enum> table;
  for (const auto value : magic_enum::enum_values<Enum>()) {
    const std::string pascalCaseName = enumNameToPascalCase(value);
    table[pascalCaseName] = value;
  }
  return table;
}

// Convert an integer to an enum with bounds checking.
// E.g. intToEnum(3, &myEnum);
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

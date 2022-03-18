// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <functional>
#include <string>
#include <unordered_map>

#include "compiler_gym/service/proto/compiler_gym_service.pb.h"

namespace compiler_gym {

struct SpaceContainsEventChecker {
  struct Context;
  using TypeIdFuncMap =
      std::unordered_map<std::string,
                         std::function<bool(const Space&, const Event&, bool, const Context&)>>;
  struct Context {
    TypeIdFuncMap typeIdFuncMap;
  };
  bool contains(const Space& space, const Event& event, bool errorOnFalse = false) const;
  void checkContains(const Space& space, const Event& event) const;
  Context& context() { return ctx_; }
  const Context& context() const { return ctx_; }

 private:
  Context ctx_;
};

SpaceContainsEventChecker makeDefaultSpaceContainsEventChecker();

}  // namespace compiler_gym

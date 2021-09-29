/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";

const ApiContext = React.createContext({
  compilerGym: {},
  session: { commandline: "", done: false, ir: "", states: [] },
  setParams: () => {},
  setSession: () => {},
  submitStep: () => {},
});

export default ApiContext;

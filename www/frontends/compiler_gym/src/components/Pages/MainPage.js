/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";
import MainNavbar from "../Navbars/MainNavbar";
import PanelsContainer from "../Sections/PanelsContainer";
import ObservationsContainer from "../Sections/ObservationsContainer";

export default function MainPage() {
  return (
    <div className="main-content">
      <MainNavbar />
      <PanelsContainer right={<ObservationsContainer />} />
    </div>
  );
}

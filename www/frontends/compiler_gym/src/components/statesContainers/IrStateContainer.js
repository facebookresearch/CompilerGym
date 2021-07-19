/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";
import classnames from "classnames";
import PropTypes from "prop-types";
import Highlight from "react-highlight";

const IrStateContainer = ({ irState, darkTheme }) => {
  return (
    <>
      <div
        className={classnames(
          "ir-state-container",
          { "dark-mode": darkTheme },
          { "light-mode": darkTheme === false }
        )}
      >
        <Highlight language="llvmir">{irState}</Highlight>
      </div>
    </>
  );
};

IrStateContainer.defaultProps = {
  irState: "",
  darkTheme: false,
};

IrStateContainer.propTypes = {
  irState: PropTypes.string,
  darkTheme: PropTypes.bool,
};

export default IrStateContainer;

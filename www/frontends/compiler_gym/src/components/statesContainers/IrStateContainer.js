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

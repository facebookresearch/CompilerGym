/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useContext, useState } from "react";
import classnames from "classnames";
import { Tabs, Tab } from "react-bootstrap";
import ApiContext from "../../context/ApiContext";
import ThemeContext from "../../context/ThemeContext";
import {
  IrStateContainer,
  InstcountsStateContainer,
  AutophaseStateContainer,
} from "../StateVisualizations";

const StateContainer = () => {
  const context = useContext(ApiContext);
  const themeContext = useContext(ThemeContext);
  const states = context.session.states;
  const [stateSelector, setStateSelector] = useState("ir");

  const last_observation = states && states[states.length - 1];
  const previous_observation = states && states[states.length - 2];

  const renderState = () => {
    switch (stateSelector) {
      case "ir":
        return (
          <IrStateContainer
            irState={context.session?.ir}
            darkTheme={themeContext.darkTheme}
          />
        );
      case "instcounts":
        return (
          <InstcountsStateContainer
            sessionStates={context.session.states}
            commandLine={context.session.commandline}
            instcount={last_observation?.instcount}
            prev_instcount ={previous_observation?.instcount || {}}
            darkTheme={themeContext.darkTheme}
          />
        );
      case "autophase":
        return (
          <AutophaseStateContainer
            sessionStates={context.session.states}
            commandLine={context.session.commandline}
            autophase={last_observation?.autophase}
            prev_authophase ={previous_observation?.autophase || {}}
            darkTheme={themeContext.darkTheme}
          />
        );
      case "programl":
        return <div>From programl</div>;
      default:
        return [];
    }
  };

  return (
    <>
      <div className="m-0 action-navbar-wrapper">
        <Tabs
          className={classnames({
            "dark_tabs": themeContext.darkTheme,
            "light_tabs": !themeContext.darkTheme,
          })}
          id="uncontrolled-tab"
          activeKey={stateSelector}
          transition={false}
          onSelect={(e) => setStateSelector(e)}
        >
          <Tab eventKey="ir" title="IR"></Tab>
          <Tab eventKey="instcounts" title="InstCounts"></Tab>
          <Tab eventKey="autophase" title="Autophase"></Tab>
        </Tabs>
      </div>
      {renderState()}
    </>
  );
};

export default StateContainer;

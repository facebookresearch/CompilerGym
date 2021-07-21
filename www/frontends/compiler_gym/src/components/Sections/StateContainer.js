/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useContext, useState } from 'react';
import { Tabs, Tab } from 'react-bootstrap';
import ApiContext from '../../context/ApiContext';
import ThemeContext from '../../context/ThemeContext';
import { IrStateContainer, InstcountsStateContainer, AutophaseStateContainer } from '../StateVisualization';

const StateContainer = () => {

    const context = useContext(ApiContext);
    const themeContext = useContext(ThemeContext)
    const envState = context.session.state
    const [stateSelector, setStateSelector] = useState("ir")

    const renderState = () => {
      switch (stateSelector) {
        case "ir":
          return <IrStateContainer irState={envState && envState.ir} darkTheme={themeContext.darkTheme}/>
        case "instcounts":
          return <InstcountsStateContainer instcount={envState && envState.instcount}/>
        case "autophase":
          return <AutophaseStateContainer autophase={envState && envState.autophase}/>
        case "programl":
          return <div>From programl</div>
        default:
          return [];
      }
    }

    return (
      <>
        <div className="m-0 action-navbar-wrapper">
          <Tabs
            id="uncontrolled-tab"
            activeKey={stateSelector}
            transition={false}
            onSelect={(e) => setStateSelector(e)}
          >
            <Tab eventKey="ir" title="IR"></Tab>
            <Tab eventKey="instcounts" title="InstCounts"></Tab>
            <Tab eventKey="autophase" title="Autophase"></Tab>
            <Tab eventKey="programl" title="ProGraML"></Tab>
          </Tabs>
        </div>
        {renderState()}
      </>
    )
}

export default StateContainer;

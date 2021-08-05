/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect } from "react";
import { Row, FormControl, Col } from "react-bootstrap";
import ActionsHeatMap from "./ActionsHeatMap";
import RewardHistoryChart from "./RewardHistoryChart";

const RewardsSection = ({ session, highlightedPoint }) => {
  const [cumulativeSum, setCumulativeSum] = useState("");
  const [toggle, setToggle] = useState(false)

  useEffect(() => {
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));
    setCumulativeSum(rewards?.reduce((a, x) => a + x, 0));
    return () => {};
  }, [session]);

  return (
    <>
      <div className="row-cumulative-reward">
        <Row className="align-items-center">
          <Col lg={10} md={10} xs={10}>
            <FormControl
              aria-describedby="basic-addon1"
              size="sm"
              type="text"
              readOnly
              value={`Cumulative Reward: ${
                cumulativeSum && cumulativeSum.toFixed(3)
              }`}
            />
          </Col>
          <Col lg={2} md={2} xs={2} className="ml-0">
            <div className="switch_box box_1">
              <input
                type="checkbox"
                className="switch_1"
                checked={toggle}
                onChange={()=> setToggle(!toggle)}
              />
            </div>
          </Col>
        </Row>
      </div>
      { toggle
        ? <ActionsHeatMap session={session} highlightedPoint={highlightedPoint}/>
        : <RewardHistoryChart session={session} /> }
    </>
  );
};
export default RewardsSection;

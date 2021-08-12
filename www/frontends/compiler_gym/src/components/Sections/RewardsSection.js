/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect, useContext } from "react";
import classnames from "classnames";
import { Row, FormControl, Col } from "react-bootstrap";
import ThemeContext from "../../context/ThemeContext";
import RewardHistoryChart from "./RewardHistoryChart";

/**
 * Renders the rewards section in the UI.
 *
 * @param {Object} session current session as a prop, needed to calculate rewards and pass to the historical chart.
 * @param {Object} highlightedPoint an object prop containing data about a node in the tree to display an action description.
 * @returns
 */
const RewardsSection = ({ session, highlightedPoint }) => {
  const { darkTheme } = useContext(ThemeContext);
  const [cumulativeSum, setCumulativeSum] = useState("");
  const [toggle, setToggle] = useState(true);

  useEffect(() => {
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));
    setCumulativeSum(rewards?.reduce((a, x) => a + x, 0));
    return () => {};
  }, [session]);

  return (
    <>
      <Row
        className={classnames(
          "row-cumulative-reward align-items-center mx-0 bg-white",
          { "bg-dark text-white": darkTheme }
        )}
      >
        <Col lg={1} md={1} xs={1} className="ml-0 mt-1">
          <div className="switch_box box_1">
            <input
              type="checkbox"
              className="switch_1"
              checked={toggle}
              onChange={() => setToggle(!toggle)}
            />
          </div>
        </Col>
        {toggle ? (
          <Col lg={4} md={4} xs={4}>
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
        ) : (
          <Col lg={4} md={4} xs={4}></Col>
        )}
        <Col lg={7} md={7} xs={7} className="text-right">
          {highlightedPoint.nodeDescription && (
            <h5>
              <span className="text-weight">Action Description: </span>
              {highlightedPoint.nodeDescription}
            </h5>
          )}
        </Col>
      </Row>
      <RewardHistoryChart session={session} />
    </>
  );
};
export default RewardsSection;

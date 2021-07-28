/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect } from "react";
import { Row, Col, Button, FormControl } from "react-bootstrap";
import RewardHistoryChart from "../Sections/RewardHistoryChart";

const RewardsNavbar = ({ session }) => {
  const [show, setShow] = useState(false);
  const [cumulativeSum, setCumulativeSum] = useState("");

  const handleCloseChart = () => setShow(false);
  const handleShowChart = () => setShow(true);

  useEffect(() => {
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));
    setCumulativeSum(rewards?.reduce((a, x) => a + x, 0));
    return () => {};
  }, [session]);

  return (
    <div className="mx-2 reward-table-wrapper">
      <Row className="align-items-center">
        <Col lg={4} md={6} xs={12} className="mt-1">
          <FormControl
            aria-describedby="basic-addon1"
            type="text"
            readOnly
            value={`Cumulative Reward: ${cumulativeSum && cumulativeSum.toFixed(3)}`}
          />
        </Col>
        <Col lg={3} md={6} xs={12} className="mt-1">
          <Button variant="primary" onClick={handleShowChart}>
            Reward History
          </Button>
        </Col>
      </Row>
      <RewardHistoryChart
        session={session}
        show={show}
        onHide={handleCloseChart}
      />
    </div>
  );
};

export default RewardsNavbar;

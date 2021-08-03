/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect, useContext, forwardRef } from "react";
import {
  FormControl,
  Col,
  InputGroup,
  Dropdown,
  Button,
} from "react-bootstrap";
import ApiContext from "../../context/ApiContext";
import RewardHistoryChart from "../Sections/RewardHistoryChart";
import LargeModal from "../Modals/LargeModal";

const CustomMenu = forwardRef(
  ({ children, style, "aria-labelledby": labeledBy }, ref) => {
    const [value, setValue] = useState("");

    return (
      <div
        ref={ref}
        style={style}
        className="cg-dropdown"
        aria-labelledby={labeledBy}
      >
        <div className="search-bar">
          <FormControl
            autoFocus
            className="my-1"
            size="sm"
            placeholder="Search..."
            onChange={(e) => setValue(e.target.value)}
            value={value}
          />
        </div>
        <ul>
          {React.Children.toArray(children).filter(
            (child) =>
              !value ||
              child.props.children
                .toLowerCase()
                .indexOf(value.toLowerCase()) !== -1
          )}
        </ul>
      </div>
    );
  }
);

const BottomControlsNavbar = ({ actionSpace, handleActionSpace, getShareLink }) => {
  const { compilerGym, session } = useContext(ApiContext);
  const [cumulativeSum, setCumulativeSum] = useState("");
  const [showChart, setShow] = useState(false);
  const [showModal, setModal] = useState(false)
  const [shareLink, setShareLink] = useState("")

  const actionSpaceOptions =
    compilerGym.actions &&
    Object.keys(compilerGym.actions).map((x, i) => i + 1);

  useEffect(() => {
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));
    setCumulativeSum(rewards?.reduce((a, x) => a + x, 0));
    setShareLink(() => getShareLink())
    return () => {};
  }, [session, getShareLink]);

  const handleCloseChart = () => setShow(false);
  const handleShowChart = () => setShow(true);
  const handleModal = () => setModal(!showModal)

  return (
    <>
      <Col lg={4} md={4} sm={12} className="mt-1 pr-lg-1">
        <InputGroup>
          <Dropdown as={InputGroup.Prepend} onSelect={handleActionSpace}>
            <Dropdown.Toggle variant="dark" id="dropdown-action-space">
              Action Space
            </Dropdown.Toggle>
            <Dropdown.Menu
              as={CustomMenu}
              style={{ margin: 0, borderRadius: "3%" }}
            >
              {actionSpaceOptions &&
                actionSpaceOptions.map((i, index) => (
                  <Dropdown.Item
                    key={index}
                    eventKey={i}
                    active={actionSpace === i.toString() ? true : false}
                  >
                    {i.toString()}
                  </Dropdown.Item>
                ))}
            </Dropdown.Menu>
          </Dropdown>
          <FormControl
            id="action-sepace-input"
            aria-describedby="basic-addon3"
            type="text"
            readOnly
            value={actionSpace}
          />
        </InputGroup>
      </Col>
      <Col lg={4} md={3} xs={12} className="mt-1 px-lg-1">
        <FormControl
          aria-describedby="basic-addon1"
          type="text"
          readOnly
          value={`Cumulative Reward: ${
            cumulativeSum && cumulativeSum.toFixed(3)
          }`}
        />
      </Col>
      <Col lg={3} md={4} xs={10} className="mt-1 pl-lg-1">
        <Button variant="primary" onClick={handleShowChart}>
          Reward History
        </Button>
      </Col>

      <Col lg={1} md={1} xs={1} className="mt-1 pl-lg-1 text-right">
        <Button variant="primary" onClick={handleModal}>
          <i className="bi bi-share-fill text-white cg-icon"></i>
        </Button>
      </Col>

      <LargeModal
        showModal={showModal}
        handleModal={handleModal}
        title={"Share"}
        content={shareLink}
      />

      <RewardHistoryChart
        session={session}
        show={showChart}
        onHide={handleCloseChart}
      />
    </>
  );
};

export default BottomControlsNavbar;

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect, useContext, forwardRef } from "react";
import {
  Form,
  FormControl,
  Col,
  Row,
  InputGroup,
  Dropdown,
  Button,
  Tooltip,
  OverlayTrigger,
} from "react-bootstrap";
import ApiContext from "../../context/ApiContext";

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

const ActionsNavbar = ({ startSession, actionSpace, handleActionSpace }) => {
  const { compilerGym, session } = useContext(ApiContext);
  const [actionsLine, setActionsLine] = useState("");
  const [dataset, setDataset] = useState("benchmark://cbench-v1");
  const [uriOptions, setUriOptions] = useState([]);
  const [datasetUri, setDatasetUri] = useState("");
  const [reward, setReward] = useState("IrInstructionCountOz");

  const benchmarkOptions =
    compilerGym.benchmarks &&
    Object.entries(compilerGym.benchmarks).map(([dataset, uri]) => ({
      dataset,
      uri,
    }));

  const actionSpaceOptions =
    compilerGym.actions &&
    Object.keys(compilerGym.actions).map((x, i) => i + 1);

  useEffect(() => {
    let lastState = session.states?.[session.states?.length - 1]
    setActionsLine(lastState?.commandline);
    return () => {};
  }, [session]);

  useEffect(() => {
    const selected =
      compilerGym.benchmarks &&
      Object.entries(compilerGym.benchmarks)
        .map(([dataset, uri]) => ({
          dataset,
          uri,
        }))
        .find((o) => o.dataset === dataset);

    setUriOptions(selected && selected.uri);
    setDatasetUri(selected && selected.uri[0]);
    return () => {};
  }, [dataset, compilerGym.benchmarks]);

  return (
    <div className="mx-2 action-navbar-wrapper">
      <Form>
        <Row className="align-items-center">
          <Col lg={5} md={6} xs={12} className="mt-1 pr-lg-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={(e) => setDataset(e)}>
                <Dropdown.Toggle variant="dark" id="dropdown-benchmark">
                  Dataset
                </Dropdown.Toggle>
                <Dropdown.Menu as={CustomMenu} style={{ margin: 0, borderRadius: "3%" }}>
                  {benchmarkOptions &&
                    benchmarkOptions.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i.dataset}
                        active={dataset === i ? true : false}
                      >
                        {i.dataset}
                      </Dropdown.Item>
                    ))}
                </Dropdown.Menu>
              </Dropdown>
              <FormControl
                aria-describedby="basic-addon1"
                type="text"
                readOnly
                value={dataset}
              />
            </InputGroup>
          </Col>
          <Col lg={3} md={6} xs={12} className="mt-1 px-lg-0">
            <InputGroup className="mb-1">
              <Dropdown
                as={InputGroup.Prepend}
                onSelect={(e) => setDatasetUri(e)}
              >
                <Dropdown.Toggle variant="dark" id="dropdown-benchmark-uri">
                  Benchmark
                </Dropdown.Toggle>
                <Dropdown.Menu as={CustomMenu} style={{ margin: 0, borderRadius: "3%" }}>
                  {uriOptions &&
                    uriOptions.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i}
                        active={datasetUri === i ? true : false}
                      >
                        {i}
                      </Dropdown.Item>
                    ))}
                </Dropdown.Menu>
              </Dropdown>
              <FormControl
                aria-describedby="basic-addon1"
                type="text"
                readOnly
                value={datasetUri}
              />
            </InputGroup>
          </Col>
          <Col lg={4} md={6} xs={12} className="mt-1 pl-lg-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={(e) => setReward(e)}>
                <Dropdown.Toggle variant="dark" id="dropdown-reward">
                  Reward
                </Dropdown.Toggle>
                <Dropdown.Menu as={CustomMenu} style={{ margin: 0, borderRadius: "3%" }}>
                  {compilerGym.rewards &&
                    compilerGym.rewards.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i}
                        active={reward === i ? true : false}
                      >
                        {i}
                      </Dropdown.Item>
                    ))}
                </Dropdown.Menu>
              </Dropdown>
              <FormControl
                id="reward-input"
                aria-describedby="basic-addon3"
                type="text"
                readOnly
                value={reward}
              />
            </InputGroup>
          </Col>
        </Row>
        <Row className="align-items-center">
          <Col sm={11} md={11} className="mt-1">
            <InputGroup className="mb-1 px-0">
                <InputGroup.Text
                  className="bg-dark"
                  id="inputGroup-sizing-sm"
                  style={{ color: "white" }}
                >
                  Actions
                </InputGroup.Text>
              <FormControl
                id="actions-input"
                type="text"
                aria-describedby="basic-addon1"
                readOnly
                defaultValue={actionsLine}
              />
            </InputGroup>
          </Col>
          <Col md={1} className="mt-1 mb-1">
            <OverlayTrigger
              placement="right"
              transition={false}
              overlay={
                <Tooltip id="button-tooltip-2">Start Session</Tooltip>
              }
            >
              {({ ref2, ...triggerHandler }) => (
                <Button
                  ref={ref2}
                  {...triggerHandler}
                  variant="success"
                  className="mr-0"
                  onClick={() => startSession(reward,`${dataset}/${datasetUri}`)}
                >
                  <i className="bi bi-play-fill"></i>
                </Button>
              )}
            </OverlayTrigger>
          </Col>
        </Row>
        <Row className="align-items-center">
          <Col lg={3} md={6} sm={12} className="mt-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={handleActionSpace}>
                <Dropdown.Toggle variant="dark" id="dropdown-action-space">
                  Action Space
                </Dropdown.Toggle>
                <Dropdown.Menu as={CustomMenu} style={{ margin: 0, borderRadius: "3%" }}>
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
        </Row>
      </Form>
    </div>
  );
};

export default ActionsNavbar;

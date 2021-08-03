/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect, useContext, forwardRef } from "react";
import classnames from "classnames";
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
  Alert,
} from "react-bootstrap";
import ApiContext from "../../context/ApiContext";
import ThemeContext from "../../context/ThemeContext";
import BottomControlsNavbar from "./BottomControlsNavbar";
import CommandLineModal from "../Modals/CommandLineModal";

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

/**
 * Represenets the navbar component inside the controls container,
 * this components takes care of the creation of a new CompilerGym environment.
 *
 * @param {function} startSession invoke API to start a new session with different specific datasets and benchmarks.
 * @param {String} actionSpace a discrete space of actions to be exposed.
 * @param {Array} actionsTaken an array of ids representing the actions selected from the tree.
 * @param {function} handleActionSpace function to update the action space.
 * @param {Object} urlParams
 * @returns
 */
const ActionsNavbar = ({
  startSession,
  actionSpace,
  actionsTaken,
  handleActionSpace,
  urlParams
}) => {
  const { darkTheme } = useContext(ThemeContext);
  const { compilerGym, session, api, setSession } = useContext(ApiContext);

  const [actionsLine, setActionsLine] = useState("");
  const [dataset, setDataset] = useState("benchmark://cbench-v1");
  const [uriOptions, setUriOptions] = useState([]);
  const [datasetUri, setDatasetUri] = useState("");
  const [reward, setReward] = useState("IrInstructionCountOz");
  const [showWarning, setShowWarning] = useState(false);
  const [showModal, setModal] = useState(false)

  const benchmarkOptions =
    compilerGym.benchmarks &&
    Object.entries(compilerGym.benchmarks).map(([dataset, uri]) => ({
      dataset,
      uri,
    }));

  /*
   * Start a new session when component mounts in the browser with URL params.
   *
   */
  useEffect(() => {
    const fetchData = async () => {
      try {
        await api.closeSession(session.session_id);
        const initSession = await api.startSession(
          urlParams.reward,
          urlParams.actions,
          `${decodeURIComponent(urlParams.dataset)}/${decodeURIComponent(urlParams.benchmark)}`
        );
        console.log(initSession);
        setSession(initSession);
        setDataset(decodeURIComponent(urlParams.dataset));
        setDatasetUri(decodeURIComponent(urlParams.benchmark));
        setReward(urlParams.reward);
      } catch (err) {
        setShowWarning(true);
      }
    };
    if (urlParams.reward && urlParams.dataset && session.session_id !== undefined) {
      fetchData();
    }

    return () => {};
  }, [
    api,
    session.session_id,
    setSession,
    urlParams.reward,
    urlParams.actions,
    urlParams.dataset,
    urlParams.benchmark,
  ]);

  useEffect(() => {
    let lastState = session.states?.[session.states?.length - 1];
    setActionsLine(lastState?.commandline);

    return () => {};
  }, [session]);

  useEffect(() => {
    let selected =
      compilerGym.benchmarks &&
      Object.entries(compilerGym.benchmarks)
        .map(([dataset, uri]) => ({
          dataset,
          uri,
        }))
        .find((o) => o.dataset === dataset);
    setUriOptions(selected?.uri);
    setDatasetUri(selected?.uri[0]);
    return () => {}
  }, [compilerGym.benchmarks, dataset]);

  const startNewSession = () => {
    startSession(reward, "-", `${dataset}/${datasetUri}`)
  };

  const getShareLink = () => {
    const dataSetEncoded = encodeURIComponent(dataset);
    const uriEncoded = encodeURIComponent(datasetUri);
    let actions = actionsTaken.map((i) => i.split(".")[0]).join(",") || "-";  // Only keep the action ids, not the depth id.
    let shareLink = `http://localhost:3000/${dataSetEncoded}/${uriEncoded}/${reward}/${actions}`;
    return shareLink;
  };

  const handleModal = () => setModal(!showModal)

  return (
    <div
      className={classnames(
        "px-2 action-navbar-wrapper",
        { "dark-mode-controls": darkTheme },
        { "": darkTheme === false }
      )}
    >
      <Form>
        <Row className="align-items-center">
          <Col lg={5} md={4} xs={12} className="mt-1 pr-lg-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={(e) => setDataset(e)}>
                <Dropdown.Toggle variant="dark" id="dropdown-benchmark">
                  Dataset
                </Dropdown.Toggle>
                <Dropdown.Menu
                  as={CustomMenu}
                  style={{ margin: 0, borderRadius: "3%" }}
                >
                  {benchmarkOptions &&
                    benchmarkOptions.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i.dataset}
                        active={dataset === i.dataset ? true : false}
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
          <Col lg={3} md={4} xs={12} className="mt-1 px-lg-0">
            <InputGroup className="mb-1">
              <Dropdown
                as={InputGroup.Prepend}
                onSelect={(e) => setDatasetUri(e)}
              >
                <Dropdown.Toggle variant="dark" id="dropdown-benchmark-uri">
                  Benchmark
                </Dropdown.Toggle>
                <Dropdown.Menu
                  as={CustomMenu}
                  style={{ margin: 0, borderRadius: "3%" }}
                >
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
          <Col lg={4} md={4} xs={12} className="mt-1 pl-lg-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={(e) => setReward(e)}>
                <Dropdown.Toggle variant="dark" id="dropdown-reward">
                  Reward
                </Dropdown.Toggle>
                <Dropdown.Menu
                  as={CustomMenu}
                  style={{ margin: 0, borderRadius: "3%" }}
                >
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
          <Col lg={10} sm={10} md={10} className="mt-1">
            <InputGroup className="mb-1">
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
          <Col lg={2} md={2} className="mt-1 mb-1 text-right">
            <OverlayTrigger
              placement="right"
              transition={false}
              overlay={<Tooltip id="button-tooltip-3">Command Line</Tooltip>}
            >
              {({ ref1, ...triggerHandler }) => (
                <Button
                  ref={ref1}
                  {...triggerHandler}
                  variant="primary"
                  className="mr-1"
                  onClick={handleModal}
                >
                  <i className="bi bi-terminal-fill text-white"></i>
                </Button>
              )}
            </OverlayTrigger>
            <OverlayTrigger
              placement="right"
              transition={false}
              overlay={<Tooltip id="button-tooltip-2">Start Session</Tooltip>}
            >
              {({ ref2, ...triggerHandler }) => (
                <Button
                  ref={ref2}
                  {...triggerHandler}
                  variant="success"
                  onClick={startNewSession}
                >
                  <i className="bi bi-cpu"></i>
                </Button>
              )}
            </OverlayTrigger>
          </Col>
        </Row>
        <Row className="align-items-center">
          <BottomControlsNavbar
            actionSpace={actionSpace}
            handleActionSpace={handleActionSpace}
            getShareLink={getShareLink}
          />
        </Row>
      </Form>
      <CommandLineModal
        showModal={showModal}
        handleModal={handleModal}
        title={"Command Line"}
        startSession={startSession}
        reward={reward}
        benchmark={`${dataset}/${datasetUri}`}
      />
      {showWarning && (
        <Alert
          variant="danger"
          className="mt-2"
          onClose={() => setShowWarning(false)}
          dismissible
        >
          <Alert.Heading>
            <span className="text-weight">Oh snap!</span> You got an error, this
            link is broken. You can still set up your own optimization steps.
          </Alert.Heading>
        </Alert>
      )}
    </div>
  );
};

export default ActionsNavbar;

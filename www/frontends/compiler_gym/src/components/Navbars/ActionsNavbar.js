/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect, useContext, forwardRef } from "react";
import classnames from "classnames";
import { useHistory, useLocation } from "react-router";
import {
  Form,
  FormControl,
  Col,
  Row,
  InputGroup,
  Dropdown,
  Alert,
  Button,
} from "react-bootstrap";
import ApiContext from "../../context/ApiContext";
import ThemeContext from "../../context/ThemeContext";
import LargeModal from "../Modals/LargeModal";
import { getCommandLineArray } from "../../utils/Helpers";

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
 * @param {function} handleResetActionsTracker updates the actions tracker state.
 * @returns
 */
const ActionsNavbar = ({
  startSession,
  actionSpace,
  actionsTaken,
  handleActionSpace,
  handleResetActionsTracker,
}) => {
  const { darkTheme } = useContext(ThemeContext);
  const { compilerGym, session, api, setSession } = useContext(ApiContext);

  const history = useHistory();
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);

  const [actionsLine, setActionsLine] = useState("");
  const [dataset, setDataset] = useState("benchmark://cbench-v1");
  const [uriOptions, setUriOptions] = useState([]);
  const [datasetUri, setDatasetUri] = useState("");
  const [reward, setReward] = useState("IrInstructionCountOz");
  const [showWarning, setShowWarning] = useState(false);
  const [showModal, setModal] = useState(false);

  const benchmarkOptions =
    compilerGym.benchmarks &&
    Object.entries(compilerGym.benchmarks).map(([dataset, uri]) => ({
      dataset,
      uri,
    }));

  const actionsList =
    compilerGym.actions &&
    Object.entries(compilerGym.actions).map(([name, action_id]) => ({
      name,
      action_id: action_id.toString(),
    }));

  // Action space as a number to show in the dropdown menu.
  const actionSpaceOptions =
    compilerGym.actions &&
    Object.keys(compilerGym.actions).map((x, i) => i + 1);

  const actionsIdsTaken = actionsTaken?.map((i) => i.split(".")[0]); // Only keep the action ids, not the depth id

  /**
   * Must run once when the component is first rendered to populate the dataset uri dropdown.
   */
  useEffect(() => {
    let selected =
      benchmarkOptions && benchmarkOptions.find((o) => o.dataset === dataset);
    setUriOptions(selected?.uri);
    setDatasetUri(selected?.uri[0]);
    return () => {};
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  /*
   * Start a new session when component mounts in the browser with URL params.
   */
  useEffect(() => {
    const fetchData = async () => {
      let selected = benchmarkOptions.find(
        (o) => o.dataset === searchParams.get("dataset")
      );
      try {
        await api.closeSession(session.session_id);
        const initSession = await api.startSession(
          searchParams.get("reward"),
          searchParams.get("actions") ?? "-",
          `${searchParams.get("dataset")}/${searchParams.get("dataset_uri")}`
        );
        console.log(initSession);
        setSession(initSession);
        setDataset(searchParams.get("dataset"));
        setDatasetUri(searchParams.get("dataset_uri"));
        setReward(searchParams.get("reward"));
        setUriOptions(selected?.uri);
      } catch (err) {
        setShowWarning(true);
      }
    };
    if (
      searchParams.get("reward") &&
      searchParams.get("dataset") &&
      searchParams.get("dataset_uri") &&
      session.session_id !== undefined
    ) {
      fetchData();
    }

    return () => {};
  }, [api, session.session_id, setSession]); // eslint-disable-line react-hooks/exhaustive-deps

  // Rerender the command line on every change in the current state of session.
  useEffect(() => {
    let lastState = session.states?.[session.states?.length - 1];
    setActionsLine(lastState?.commandline);

    return () => {};
  }, [session]);

  const handleNewDataset = (e) => {
    let selected = benchmarkOptions.find((o) => o.dataset === e);
    setDataset(e);
    setUriOptions(selected?.uri);
    setDatasetUri(selected?.uri[0]);
    startSession(
      reward,
      actionsIdsTaken.length ? actionsIdsTaken : "-",
      `${e}/${selected?.uri[0]}`
    );
    searchParams.set("dataset", e);
    searchParams.set("dataset_uri", selected?.uri[0]);
    searchParams.set("reward", reward);
    history.push({ ...location, search: searchParams.toString() });
  };

  const handleDatasetUri = (e) => {
    setDatasetUri(e);
    startSession(
      reward,
      actionsIdsTaken.length ? actionsIdsTaken : "-",
      `${dataset}/${e}`
    );
    searchParams.set("dataset", dataset);
    searchParams.set("dataset_uri", e);
    searchParams.set("reward", reward);
    history.push({ ...location, search: searchParams.toString() });
  };

  const handleRewardSelect = (e) => {
    setReward(e);
    startSession(
      e,
      actionsIdsTaken.length ? actionsIdsTaken : "-",
      `${dataset}/${datasetUri}`
    );
    searchParams.set("dataset", dataset);
    searchParams.set("dataset_uri", datasetUri);
    searchParams.set("reward", e);
    history.push({ ...location, search: searchParams.toString() });
  };

  // Append current url params into a string to generata a link.
  const getShareLink = () => {
    let shareLink = `${window.location.href}${location.search}`;
    return shareLink;
  };

  /**
   * Run a set of actions when clicking on the comand line text input.
   * @param {*} e listens to an event on keyPress.
   */
  const runCommandLine = async (e) => {
    if (e.key === "Enter") {
      try {
        let actionsTaken = getCommandLineArray(actionsLine, actionsList);
        await startSession(
          reward,
          actionsTaken.length ? actionsTaken : "-",
          `${dataset}/${datasetUri}`
        );
        searchParams.set("dataset", dataset);
        searchParams.set("dataset_uri", datasetUri);
        searchParams.set("reward", reward);
        searchParams.set("actions", actionsTaken.join(","));
        history.replace({ ...location, search: searchParams.toString() });
      } catch (error) {
        console.log(error);
      }
    }
  };

  // Close warning popup and start a new session.
  const handleWarningAlert = async () => {
    try {
      setShowWarning(false);
      history.push("/");
      await api.startSession(reward, "-", `${dataset}/${datasetUri}`);
      handleResetActionsTracker();
    } catch (error) {
      console.log(error);
    }
  };

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
          <Col lg={3} md={3} xs={12} className="mt-1 pr-lg-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={handleNewDataset}>
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
          <Col lg={3} md={3} xs={12} className="mt-1 pl-lg-0 pr-lg-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={handleDatasetUri}>
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
          <Col lg={3} md={3} xs={12} className="mt-1 pl-lg-0 pr-lg-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={handleRewardSelect}>
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
          <Col lg={3} md={3} sm={12} className="mt-1 pl-lg-0">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={handleActionSpace}>
                <Dropdown.Toggle variant="dark" id="dropdown-action-space">
                  Actions
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
        </Row>
        <Row className="align-items-center">
          <Col lg={11} md={11} xs={11} className="my-1">
            <InputGroup>
              <InputGroup.Text
                className="bg-dark"
                id="inputGroup-sizing-sm"
                style={{ color: "white" }}
              >
                Command
              </InputGroup.Text>
              <FormControl
                id="actions-input"
                type="text"
                aria-describedby="basic-addon1"
                className={classnames("", {
                  "bg-darker text-white": darkTheme,
                })}
                value={actionsLine}
                onChange={(e) => setActionsLine(e.target.value)}
                onKeyPress={runCommandLine}
              />
            </InputGroup>
          </Col>
          <Col lg={1} md={1} xs={1} className="my-1 pl-lg-1 text-right">
            <Button variant="primary" onClick={() => setModal(!showModal)}>
              <i className="bi bi-share-fill text-white cg-icon"></i>
            </Button>
          </Col>
        </Row>
      </Form>
      {showWarning && (
        <Alert
          variant="danger"
          className="mt-2"
          dismissible
          onClose={handleWarningAlert}
        >
          <Alert.Heading>
            <span className="text-weight">Oh snap!</span> You got an error, this
            link is broken. You can still set up your own optimization steps.
          </Alert.Heading>
        </Alert>
      )}
      <LargeModal
        showModal={showModal}
        handleModal={() => setModal(!showModal)}
        title={"Share"}
        getShareLink={getShareLink}
      />
    </div>
  );
};

export default ActionsNavbar;

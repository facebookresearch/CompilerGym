/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext, useEffect } from "react";
import { useHistory, useLocation } from "react-router";
import { Spinner } from "react-bootstrap";
import ApiContext from "../../context/ApiContext";
import { makeSessionTreeData } from "../../utils/Helpers";
import ActionsNavbar from "../Navbars/ActionsNavbar";
import SearchTree from "./SearchTree";
import RewardsSection from "./RewardsSection";
import ActionsDict from "../../utils/ActionsDict";

const ControlsContainer = () => {
  const { compilerGym, session, params, api, setSession } =
    useContext(ApiContext);
  const history = useHistory();
  const location = useLocation();
  const searchParams = new URLSearchParams(location.search);

  const [actionSpace, setActionSpace] = useState(30);
  const [treeData, setTreeData] = useState({});
  const [highlightedPoint, setHighlightedPoint] = useState({});
  const [focusedNode, setFocusedNode] = useState(0);
  const [actionsTracker, setActionsTracker] = useState({
    activeNode: "x",
    actionsTaken: [],
    layer: 1,
  });
  const [locationKeys, setLocationKeys] = useState([]);
  const [isLoading, setIsLoading] = useState(false)

  const children =
    compilerGym.actions &&
    Object.entries(compilerGym.actions).map(([name, action_id]) => ({
      name,
      action_id: action_id.toString(),
      description:
        ActionsDict.find((i) => i.Action === name)?.Description || "",
      children: [],
    }));

  /**
   * Check whether a set of actions is passed as params in the URL, if yes, updates the DOM
   * to render a search tree with a specific state. Otherwise, renders a clean tree with only
   * one level depth, one set of children and root as active node.
   */
  useEffect(() => {
    let urlIds = searchParams.get("actions")?.split(",") ?? [];
    let actionsTaken = urlIds.map((o, i) => `${o}.${i + 1}`);

    if (urlIds.length && session.states) {
      setTreeData(
        makeSessionTreeData(session.states, children, session.commandline)
      );
      setActionsTracker({
        activeNode: `${urlIds[urlIds.length - 1]}.${urlIds.length}`,
        actionsTaken: actionsTaken,
        layer: urlIds.length + 1,
      });
    } else if (!urlIds.length && session.states) {
      setTreeData(
        makeSessionTreeData(session.states, children, session.commandline)
      );
    }
    return () => {};
  }, [session.states]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * Detects back and forward buttons in the browser, it allows to keep track
   * of the session and observation states.
   */
  useEffect(() => {
    return history.listen((location) => {
      if (history.action === "PUSH") {
        setLocationKeys([location.key]);
      }
      if (history.action === "POP") {
        if (locationKeys[1] === location.key) {
          setLocationKeys(([_, ...keys]) => keys);
        } else {
          setLocationKeys((keys) => [location.key, ...keys]);
          window.location.reload();
        }
      }
    });
  }, [locationKeys]); // eslint-disable-line react-hooks/exhaustive-deps

  /**
   * This functions makes an API call to replay a session with new parameters
   *
   * @param {String} dataset a dataset string.
   * @param {String} datasetUri a string representing the uri of a dataset.
   * @param {String} reward the name of the reward signal to use.
   * @param {Array} actions an optional array of action ids.
   */
  const startNewSession = async (dataset, datasetUri, reward, actions) => {
    const actionsIdsTaken = actionsTracker.actionsTaken?.map(
      (i) => i.split(".")[0]
    ); // Only keep the action ids, not the depth id
    try {
      setIsLoading(true)
      if (actions !== undefined) {
        const result = await api.getActions(`${dataset}/${datasetUri}`, reward, actions, "1");
        console.log(result);
        setSession(result);
        setActionSpace(30);
        let actionsTaken = actions.map((o, i) => `${o}.${i + 1}`);
          setActionsTracker({
            activeNode: actionsTaken[actionsTaken.length - 1],
            actionsTaken: actionsTaken,
            layer: actionsTaken.length + 1,
          });
      } else {
        const result = await api.getActions(`${dataset}/${datasetUri}`, reward, actionsIdsTaken.length ? actionsIdsTaken : "", actionsIdsTaken.length ? "1" : null );
        console.log(result);
        setSession(result);
        setActionSpace(30);
        if (actionsIdsTaken.length) {
          let actionsTaken = actionsIdsTaken.map((o, i) => `${o}.${i + 1}`);
          setActionsTracker({
            activeNode: actionsTaken[actionsTaken.length - 1],
            actionsTaken: actionsTaken,
            layer: actionsTaken.length + 1,
          });
        } else {
          setActionsTracker({
            activeNode: "x",
            actionsTaken: [],
            layer: 1,
          });
        }
      }
      setIsLoading(false)
    } catch (err) {
      console.log(err);
    }
  };

  /**
   * Invokes the API to take a step in the current session and update the tree.
   * It also updates the url parameters to keep the state of current session.
   *
   * @param {Number} stepID receives an action id.
   */
  const submitStep = async (stepID) => {
    let urlActions = actionsTracker.actionsTaken
      .map((i) => i.split(".")[0])
      .join(",");
    let newActionIds = [
      ...actionsTracker.actionsTaken.map((i) => i.split(".")[0]),
      stepID,
    ];
    try {
      const response = await api.getActions(
        `${params.dataset}/${params.datasetUri}`,
        params.reward,
        newActionIds,
        "1"
      );
      setSession(response);
      setTreeData(
        makeSessionTreeData(response.states, children, response.commandline)
      );
      searchParams.set("actions", `${urlActions},${stepID}`);
      history.push({ ...location, search: searchParams.toString() });
    } catch (err) {
      console.log(err);
    }
  };

  /**
   * Invokes the API to undo a number of steps on the current session and update the tree.
   *
   */
  const undoStep = async () => {
    let newActionIds = actionsTracker.actionsTaken
      .map((i) => i.split(".")[0])
      .slice(0, -1);
    let urlActions = newActionIds.join(",");
    try {
      const response = await api.getActions(
        `${params.dataset}/${params.datasetUri}`,
        params.reward,
        newActionIds,
        "1"
      );
      setSession(response);
      setTreeData(
        makeSessionTreeData(response.states, children, response.commandline)
      );
      searchParams.set("actions", urlActions);
      history.push({ ...location, search: searchParams.toString() });
    } catch (err) {
      console.log(err);
    }
  };

  /**
   * A function that makes an API call to replay an episode of actions
   * when a node between the root and last children is clicked in the tree.
   * It also updates the url parameters to keep the history of current session.
   *
   * @param {Array} stepsIDs an array of action ids.
   */
  const replicateSteps = async (stepsIDs) => {
    let newActionsTaken = stepsIDs.map((o, i) => `${o}.${i + 1}`);
    try {
      const response = await api.getActions(
        `${params.dataset}/${params.datasetUri}`,
        params.reward,
        stepsIDs,
        "1"
      );
      setSession(response);
      setTreeData(
        makeSessionTreeData(response.states, children, response.commandline)
      );
      setActionsTracker({
        activeNode: newActionsTaken[newActionsTaken.length - 1],
        actionsTaken: newActionsTaken,
        layer: newActionsTaken.length + 1,
      });
      searchParams.set("actions", stepsIDs.join(","));
      history.push({ ...location, search: searchParams.toString() });
    } catch (error) {
      console.log(error);
    }
  };

  /**
   * The handleNodeClick is an asynchronous function and has three scenarios:
   * 1. when a node has no children, and is part of the last children, it creates an array new children with unique id,
   * 2. when a node is a sibling of the last parent component. Undo the last action.
   * 3. when a node is between the root and the last children, it gets activated and switched its siblings off.
   *
   * @param {Object} nodeDatum object containg data of the node.
   * @returns
   */
  const handleNodeClick = async (nodeDatum) => {
    let nodeActionID = nodeDatum.action_id.split(".")[0];
    let nodeDepth = nodeDatum.__rd3t.depth;

    if (nodeDatum !== undefined && nodeDepth !== 0) {
      try {
        // Verifies it is one of the last children.
        if (nodeDepth === session.states.length) {
          await submitStep(nodeActionID);
          setActionsTracker({
            activeNode: nodeDatum.action_id,
            actionsTaken: [...actionsTracker.actionsTaken, nodeDatum.action_id],
            layer: actionsTracker.layer + 1,
          });
        } else if (nodeDepth === session.states.length - 1 && nodeDepth > 1) {
          await undoStep();
          setActionsTracker({
            activeNode:
              actionsTracker.actionsTaken.length > 1
                ? actionsTracker.actionsTaken[
                    actionsTracker.actionsTaken.length - 2
                  ]
                : "x",
            actionsTaken: actionsTracker.actionsTaken.slice(0, -1),
            layer: actionsTracker.layer - 1,
          });
        } else if (nodeDepth < session.states.length && nodeDepth > 0) {
          let actionIds = actionsTracker.actionsTaken
            .map((i) => i.split(".")[0])
            .slice(); // get a copy of actionsTaken
          actionIds.splice(
            nodeDepth - 1,
            actionIds.length - nodeDepth + 1,
            nodeActionID
          ); // modify the array of actions adding the node clicked and removing ones on the right side of array.
          await replicateSteps(actionIds);
        } else {
          return;
        }
      } catch (err) {
        console.log(err);
      }
    }
    return;
  };

  /**
   * Recursive function to update number of links shown on active node.
   *
   * @param {Object} tree receives the current state of the tree data as a hierarchical object.
   * @param {String} activeNode receives a string that represents the id of actiove node.
   * @param {Number} limit represents the number of children to display.
   * @returns
   */
  const updateNode = (tree, activeNode, limit) => {
    if (tree.action_id === activeNode) {
      return { ...tree, children: children.slice(0, limit) };
    }
    if (tree.children !== undefined) {
      tree.children.forEach((i) => {
        if (i.action_id === activeNode) {
          i.active = true;
          i.children = children.slice(0, limit).map((o) => {
            return {
              name: o.name,
              description:
                ActionsDict.find((i) => i.Action === o.name)?.Description || "",
              action_id: `${o.action_id}.${actionsTracker.layer}`,
              children: [],
            };
          });
        } else {
          updateNode(i, activeNode, limit);
        }
      });
      return { name: "root", action_id: "x", children: tree.children };
    }
    return;
  };

  /**
   * Controls the number of actions[nodes] to show as children in the tree.
   *
   * @param {Event} e an event parameter containing input from dropdown.
   */
  const handleActionSpace = (e) => {
    setActionSpace(e);
    setTreeData(updateNode(treeData, actionsTracker.activeNode, e));
  };

  const handleResetActionsTracker = () => {
    setActionsTracker({ activeNode: "x", actionsTaken: [], layer: 1 });
  };

  const handleMouseOverTree = (nodeData) => {
    let nodeDescription =
      ActionsDict.find((i) => i.Action === nodeData.name)?.Description || "";
    if (nodeData.active) {
      setHighlightedPoint({
        point: nodeData.__rd3t.depth,
        selected: true,
        nodeDescription: nodeDescription,
      });
    } else {
      setHighlightedPoint({
        ...highlightedPoint,
        point: nodeData.__rd3t.depth,
        nodeDescription: nodeDescription,
      });
    }
  };

  const handleMouseOutTree = (nodeData) => {
    if (nodeData.active) {
      setHighlightedPoint({
        point: nodeData.__rd3t.depth,
        selected: false,
        nodeDescription: "",
      });
    } else {
      setHighlightedPoint({
        ...highlightedPoint,
        nodeDescription: "",
        point: null,
      });
    }
  };

  /**
   * a setter that listens to a click on the historial reward chart.
   * Implemented in the lookup of a specific node
   *
   * @param {object} dataPoint an object representing a point from the historical chart.
   */
  const handleClickOnChart = (dataPoint) => {
    setFocusedNode(dataPoint.index);
  };

  return (
    <div>
      <ActionsNavbar
        actionSpace={actionSpace}
        startSession={startNewSession}
        setIsLoading={setIsLoading}
        handleActionSpace={handleActionSpace}
        handleResetActionsTracker={handleResetActionsTracker}
      />
      {isLoading ? (
        <div className="spinner-wrap">
          <Spinner
            animation="border"
            role="status"
            aria-hidden="true"
          />
        </div>
      ) : (
        <SearchTree
          actionSpace={actionSpace}
          treeData={treeData}
          layer={actionsTracker.layer}
          focusedNode={focusedNode}
          handleNodeClick={handleNodeClick}
          handleMouseOverTree={handleMouseOverTree}
          handleMouseOutTree={handleMouseOutTree}
        />
      )}
      <RewardsSection
        session={session}
        highlightedPoint={highlightedPoint}
        handleClickOnChart={handleClickOnChart}
      />
    </div>
  );
};

export default ControlsContainer;

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext, useEffect } from "react";
import { useParams, useHistory } from "react-router";
import ApiContext from "../../context/ApiContext";
import { makeSessionTreeData } from "../../utils/Helpers";
import ActionsNavbar from "../Navbars/ActionsNavbar";
import SearchTree from "./SearchTree";
import RewardsSection from "./RewardsSection";

const ControlsContainer = () => {
  const { compilerGym, session, api, setSession } = useContext(ApiContext);
  const urlParams = useParams();
  const history = useHistory();

  const [actionSpace, setActionSpace] = useState(30);
  const [treeData, setTreeData] = useState({});
  const [layer, setLayer] = useState(1);
  const [activeNode, setActiveNode] = useState("x");
  const [actionsTaken, setActionsTaken] = useState([]);
  const [highlightedPoint, setHighlightedPoint] = useState({});

  const children =
    compilerGym.actions &&
    Object.entries(compilerGym.actions).map(([name, action_id]) => ({
      name,
      action_id: action_id.toString(),
      children: [],
    }));

  /**
   * Check whether a set of actions is passed as params in the URL, if yes, updates the DOM
   * to render a search tree with a specific state. Otherwise, renders a clean tree with only
   * one level depth and one set of children.
   */
  useEffect(() => {
    let children =
      compilerGym.actions &&
      Object.entries(compilerGym.actions).map(([name, action_id]) => ({
        name,
        action_id: action_id.toString(),
        children: [],
      }));
    let urlIds = urlParams.actions?.split(",") || [];
    let actionsTaken = urlIds.map((o, i) => `${o}.${i + 1}`);
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));

    if (urlIds.length > 0 && urlIds.length === rewards?.length - 1) {
      setTreeData(makeSessionTreeData(session.states, children));
      setLayer(urlIds.length + 1);
      setActionsTaken(actionsTaken);
      setActiveNode(`${urlIds[urlIds.length - 1]}.${urlIds.length}`);
    } else if (session.states) {
      setTreeData(makeSessionTreeData(session.states, children));
    }
    return () => {};
  }, [compilerGym.actions, urlParams.actions, session.states]);

  /**
   * This functions makes an API call to close current session and start a new session with new parameters
   *
   * @param {String} reward Takes the type of reward to initialize the new session with.
   * @param {Array} actions array with ids of actions.
   * @param {String} newBenchmark Takes the benchamark to initialize a new session.
   */
  const startNewSession = (reward, actions, newBenchmark) => {
    history.push("/");
    api.closeSession(session.session_id).then(
      (res) => {
        api.startSession(reward, actions, newBenchmark).then(
          (result) => {
            setSession(result);
            setActionSpace(30);
            if (actions !== "-" && actions.length) {
              let actionsTaken = actions.map((o, i) => `${o}.${i + 1}`);
              setActionsTaken(actionsTaken);
              setActiveNode(actionsTaken[actionsTaken.length - 1]);
            } else {
              setActionsTaken([]);
              setActiveNode("x");
            }
          },
          (error) => {
            console.log(error);
          }
        );
      },
      (error) => {
        console.log(error);
      }
    );
  };

  /**
   * This function invokes the API to take a step in the current session and update the tree.
   *
   * @param {Array} stepsIDs receives an array of action ids.
   */
  const submitStep = async (stepsIDs) => {
    try {
      const response = await api.getSteps(session.session_id, stepsIDs);
      setSession({
        ...session,
        states: [...session.states, ...response.states],
      });
      setTreeData(
        makeSessionTreeData([...session.states, ...response.states], children)
      );
    } catch (err) {
      console.log(err);
    }
  };

  /**
   * This function invokes the API to undo a number of steps on the current session and update the tree.
   *
   * @param {Number} n the number of steps to remove.
   */
  const undoStep = async (n) => {
    let currentSteps = session.states;
    try {
      await api.undoStep(session.session_id, n);
      setSession({ ...session, states: currentSteps.slice(0, -n) });
      setTreeData(makeSessionTreeData(currentSteps.slice(0, -n), children));
    } catch (err) {
      console.log(err);
    }
  };

  /**
   * This function makes two API calls to undo a number of actions and replay episode.
   * when a node between root and last children is clicked in the tree.
   *
   * @param {Number} n the number of steps to remove.
   * @param {Array} stepsIDs an array of action ids.
   */
  const replicateSteps = async (n, stepsIDs) => {
    try {
      await api.undoStep(session.session_id, n);
      const response = await api.getSteps(session.session_id, stepsIDs);
      setSession({
        ...session,
        states: [session.states[0], ...response.states],
      });
      setTreeData(
        makeSessionTreeData([session.states[0], ...response.states], children)
      );
      setActionsTaken(stepsIDs.map((o, i) => `${o}.${i + 1}`));
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
          await submitStep([nodeActionID]);
          setLayer(layer + 1);
          setActiveNode(nodeDatum.action_id);
          setActionsTaken([...actionsTaken, nodeDatum.action_id]);
        } else if (nodeDepth === session.states.length - 1 && nodeDepth > 1) {
          await undoStep(1);
          setLayer(layer - 1);
          setActiveNode(
            actionsTaken.length > 1
              ? actionsTaken[actionsTaken.length - 2]
              : "x"
          );
          setActionsTaken(actionsTaken.slice(0, -1));
        } else if (nodeDepth < session.states.length && nodeDepth > 0) {
          let actions = actionsTaken.map((i) => i.split(".")[0]).slice(); // Get a copy of actionsTaken
          actions.splice(nodeDepth - 1, 1, nodeActionID); // modify the array of actions adding the node clicked and replacing old node.
          await replicateSteps(actionsTaken.length, actions);
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
              action_id: `${o.action_id}.${layer}`,
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

  const handleActionSpace = (e) => {
    setActionSpace(e);
    setTreeData(updateNode(treeData, activeNode, e));
  };

  const handleMouseOverTree = (nodeData) => {
    if (nodeData.active) {
      setHighlightedPoint({ point: nodeData.__rd3t.depth, selected: true });
    }
  };

  const handleMouseOutTree = (nodeData) => {
    if (nodeData.active) {
      setHighlightedPoint({ point: nodeData.__rd3t.depth, selected: false });
    }
  };

  return (
    <div>
      <ActionsNavbar
        actionSpace={actionSpace}
        actionsTaken={actionsTaken}
        urlParams={urlParams}
        startSession={startNewSession}
        handleActionSpace={handleActionSpace}
      />
      <SearchTree
        actionSpace={actionSpace}
        treeData={treeData}
        layer={layer}
        handleNodeClick={handleNodeClick}
        handleMouseOverTree={handleMouseOverTree}
        handleMouseOutTree={handleMouseOutTree}
      />
      <RewardsSection session={session} highlightedPoint={highlightedPoint} />
    </div>
  );
};

export default ControlsContainer;

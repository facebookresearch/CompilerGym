/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext, useEffect } from "react";
import { useParams, useHistory } from "react-router";
import ApiContext from "../../context/ApiContext";
import { makeTreeDataFromURL } from "../../utils/Helpers"
import ActionsNavbar from "../Navbars/ActionsNavbar";
import SearchTree from "./SearchTree";
import RewardsSection from "./RewardsSection";

const ControlsContainer = () => {
  const { compilerGym, session, api, setSession } = useContext(ApiContext);
  const urlParams = useParams();
  const history = useHistory();

  const [actionSpace, setActionSpace] = useState(30);
  const [actionsList, setActionsList] = useState([]);
  const [treeData, setTreeData] = useState({});
  const [layer, setLayer] = useState(1);
  const [activeNode, setActiveNode] = useState("x");
  const [actionsTaken, setActionsTaken] = useState([]);
  const [highlightedPoint, setHighlightedPoint] = useState({})

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
    let actionsTaken = urlIds.map((o,i) => `${o}.${i+1}`)
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));

    if (urlIds.length > 0 && urlIds.length === rewards?.length - 1) {
      setActionsList(children);
      setTreeData(makeTreeDataFromURL(urlIds, children, rewards));
      setLayer(urlIds.length+1)
      setActionsTaken(actionsTaken)
      setActiveNode(`${urlIds[urlIds.length - 1]}.${urlIds.length}`)
    } else {
      setActionsList(children);
      setTreeData({
        name: "root",
        action_id: "x",
        children: children?.slice(0, 30),
      });
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
    const children =
      compilerGym.actions &&
      Object.entries(compilerGym.actions).map(([name, action_id]) => ({
        name,
        action_id: action_id.toString(),
        children: [],
      }));
    history.push("/");
    api.closeSession(session.session_id).then(
      (res) => {
        api.startSession(reward, actions, newBenchmark).then(
          (result) => {
            setSession(result);
            setTreeData({
              name: "root",
              action_id: "x",
              children: children?.slice(0, 30),
            });
            setActionsList(children);
            setActionSpace(30);
            setLayer(1);
            setActionsTaken([]);
            setActiveNode("x");
            if (actions !== "-" && actions.length) {
              setActionsTaken(actions?.map((o,i) => `${o}.${i+1}`))
            } else {
              setActionsTaken([])
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
   * Recursive function to create new node + layer when user clicks on a node.
   *
   * @param {Array} arr Receives the children array in the root node.
   * @param {String} actionID Receives the tree action_id of the node.
   * @param {String} reward takes the reward from api call.
   * @returns
   */
  const createNode = (arr, actionID, reward) => {
    if (arr !== undefined) {
      arr.forEach((i) => {
        if (i.action_id === actionID) {
          i.active = true;
          i.reward = reward;
          i.children = actionsList.slice(0, actionSpace).map((o) => {
            return {
              name: o.name,
              action_id: `${o.action_id}.${layer + 1}`,
              children: [],
            };
          });
        } else {
          createNode(i.children, actionID, reward);
        }
      });
      return { name: "root", action_id: "x", children: arr };
    }
    return;
  };

  /**
   * Recursive function to delete a node in nested child object.
   *
   * @param {Array} arr Receives the children array in the root node.
   * @param {String} actionID Receives the tree action_id of the node.
   * @returns
   */
  const deleteNode = (arr, actionID) => {
    if (arr !== undefined) {
      arr.forEach((i) => {
        if (i.action_id === actionID) {
          i.active = false;
          i.children = [];
        } else {
          deleteNode(i.children, actionID);
        }
      });
      return { name: "root", action_id: "x", children: arr };
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
      return { ...tree, children: actionsList.slice(0, limit) };
    }
    if (tree.children !== undefined) {
      tree.children.forEach((i) => {
        if (i.action_id === activeNode) {
          i.active = true;
          i.children = actionsList.slice(0, limit).map((o) => {
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

  const submitStep = async (stepID) => {
    try {
      const response = await api.getSteps(session.session_id, stepID);
      setSession({
        ...session,
        states: [...session.states, ...response.states],
      });
      const stepReward = response.states[0].reward.toFixed(3);
      return stepReward;
    } catch (err) {
      console.log(err);
    }
  };

  const undoStep = async (n) => {
    let currentSteps = session.states;
    try {
      const result = await api.undoStep(session.session_id, n);
      let actionToUndo = currentSteps.indexOf(result);
      currentSteps.splice(actionToUndo, 1);
      setSession({ ...session, states: currentSteps });
    } catch (err) {
      console.log(err);
    }
  };

  /**
   * The handleNodeClick is an asynchronous function and has two scenarios:
   * 1. when a node has no children, adds the node as activeNode it creates an array of children with unique id,
   *    the action id + the depth e.g. 12.2, in which 12 represents the action ID and 2 represents the layer.
   * 2. when a node has children, it removes its children and updates the activeNode to the previous node.
   */

  const handleNodeClick = async (nodeDatum) => {
    let nodeActionID = nodeDatum.action_id.split(".")[0];
    let nodeDepth = nodeDatum.__rd3t.depth;

    if (nodeDatum !== undefined && nodeDepth !== 0) {
      try {
        if (nodeDepth === layer) {
          let reward = await submitStep(nodeActionID);
          setLayer(layer + 1);
          setActiveNode(nodeDatum.action_id);
          setActionsTaken([...actionsTaken, nodeDatum.action_id]);

          setTreeData(
            createNode(treeData.children, nodeDatum.action_id, reward)
          );
        } else if (
          nodeDepth === layer - 1 &&
          actionsTaken.includes(nodeDatum.action_id)
        ) {
          undoStep(1);
          setLayer(layer - 1);
          setActiveNode(
            actionsTaken.length > 1
              ? actionsTaken[actionsTaken.length - 2]
              : "x"
          );
          setActionsTaken((prev) =>
            prev.filter((i) => i !== nodeDatum.action_id)
          );
          setTreeData(deleteNode(treeData.children, nodeDatum.action_id));
        }
      } catch (err) {
        console.log(err);
      }
    }
    return;
  };

  const handleActionSpace = (e) => {
    setActionSpace(e);
    setTreeData(updateNode(treeData, activeNode, e));
  };

  const handleMouseOverTree = (nodeData) => {
    if (nodeData.active) {
      setHighlightedPoint({point: nodeData.__rd3t.depth, selected: true})
    }
  }

  const handleMouseOutTree = (nodeData) => {
    if (nodeData.active) {
      setHighlightedPoint({point: nodeData.__rd3t.depth, selected: false})
    }
  }

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
      <RewardsSection
        session={session}
        highlightedPoint={highlightedPoint}
      />
    </div>
  );
};

export default ControlsContainer;

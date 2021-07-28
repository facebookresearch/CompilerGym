/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, {
  useState,
  useContext,
  useEffect,
  forwardRef,
  useRef,
} from "react";
import classnames from "classnames";
import { Row, Col, InputGroup, FormControl, Dropdown } from "react-bootstrap";
import ApiContext from "../../context/ApiContext";
import ThemeContext from "../../context/ThemeContext";
import RewardsNavbar from "../Navbars/RewardsNavbar";
import Tree from "react-d3-tree";

const DropdownMenu = forwardRef(
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
              !value || child.props.children.toLowerCase().startsWith(value)
          )}
        </ul>
      </div>
    );
  }
);
/**
 * Renders a node component.
 *
 * @param {Object} nodeDatum Receieves an object with node data.
 * @param {Function} handleNodeClick Function to be called when a node is clicked.
 * @param {Number} layer Takes the current active layer.
 * @returns {JSX} JSX element representing a node.
 */

const renderSvgNode = ({ nodeDatum, handleNodeClick, layer }) => {
  const display =
    nodeDatum.__rd3t.depth === 0 ||
    nodeDatum.__rd3t.depth === layer ||
    nodeDatum.active
      ? true
      : false;
  const foreignObjectProps = { width: 115, height: 200, x: 20, y: 20 };

  return (
    <g style={{ visibility: display ? "visible" : "hidden" }}>
      <circle
        r="5"
        x="0"
        fill={nodeDatum.active ? "#2dce89" : "white"}
        strokeWidth="1"
        onClick={() => handleNodeClick(nodeDatum)}
      />
      {nodeDatum.active && (
        <foreignObject {...foreignObjectProps}>
          <div className="active-node-info">
            {nodeDatum.children && (
              <h5>
                <span className="text-weight">Reward:</span> {nodeDatum.reward}
              </h5>
            )}
          </div>
        </foreignObject>
      )}
      <text
        strokeWidth="0"
        x="8"
        dy="5"
        onClick={() => handleNodeClick(nodeDatum)}
      >
        {nodeDatum.name}
      </text>
    </g>
  );
};

const SearchTree = () => {
  const { compilerGym, session, api, setSession } = useContext(ApiContext);
  const { darkTheme } = useContext(ThemeContext);

  const [actionSpace, setActionSpace] = useState(30);
  const [treeData, setTreeData] = useState({});
  const [layer, setLayer] = useState(1);
  const [nodeSize, setNodeSize] = useState({ x: 300, y: 20 });
  const [actionsTaken, setActionsTaken] = useState([]);

  const [actionsList, setActionsList] = useState([]);
  const [activeNode, setActiveNode] = useState("x");

  const treeWindow = useRef();

  const actionSpaceOptions =
    compilerGym.actions &&
    Object.keys(compilerGym.actions).map((x, i) => i + 1);

  useEffect(() => {
    const children =
      compilerGym.actions &&
      Object.entries(compilerGym.actions).map(([name, action_id]) => ({
        name,
        action_id: action_id.toString(),
        children: [],
      }));
    setActionsList(children);
    setTreeData({
      name: "root",
      action_id: "x",
      children: children?.slice(0, 30),
    });
  }, [compilerGym.actions]);

  useEffect(() => {
    setNodeSize({ x: actionSpace > 100 ? 700 : 300, y: 20 });
    return () => {};
  }, [actionSpace]);

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
      return stepReward
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

  const getDynamicPathClass = ({ source, target }) => {
    if (!target.children && target.depth < layer) return "link__to-leaf";
  };

  return (
    <>
      <div className="mx-2 searchtree-nav-wrapper">
        <Row className="align-items-center">
          <Col md={4} sm={12}>
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={handleActionSpace}>
                <Dropdown.Toggle variant="dark" id="dropdown-action-space">
                  Action Space
                </Dropdown.Toggle>
                <Dropdown.Menu
                  as={DropdownMenu}
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
      </div>
      <RewardsNavbar session={session} />
      <div
        ref={treeWindow}
        className={classnames(
          "search-tree-container",
          { "dark-mode-tree": darkTheme },
          { "light-mode-tree": darkTheme === false }
        )}
      >
        <Tree
          data={treeData}
          nodeSize={nodeSize}
          translate={{ x: 10, y: treeWindow.current?.clientHeight / 3 || 10 }}
          pathClassFunc={getDynamicPathClass}
          renderCustomNodeElement={(rd3tProps) =>
            renderSvgNode({ ...rd3tProps, handleNodeClick, layer })
          }
        />
      </div>
    </>
  );
};

export default SearchTree;

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
import { Col, Form, InputGroup, FormControl, Dropdown } from "react-bootstrap";
import ApiContext from "../context/ApiContext";
import ThemeContext from "../context/ThemeContext";
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
 * @param {Function} handleNodeClick Function to call when a node is clecked.
 * @returns
 */
const renderSvgNode = ({ nodeDatum, handleNodeClick }) => {
  return (
    <g>
      <circle
        r="5"
        x="0"
        fill="white"
        strokeWidth="1"
        onClick={() => handleNodeClick(nodeDatum)}
      />
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

const ActionsSearchTree = () => {
  const { compilerGym, session, api, submitStep, setSession } = useContext(ApiContext);
  const { darkTheme } = useContext(ThemeContext);

  const [actionSpace, setActionSpace] = useState(15);
  const [treeData, setTreeData] = useState({});
  const [treeGlobalData, setTreeGlobalData] = useState({});
  const [layer, setLayer] = useState(1);
  const [nodeSize, setNodeSize] = useState({ x: 300, y: 20 });
  const [actionsTaken, setActionsTaken] = useState([]);

  const treeWindow = useRef();

  const actionSpaceOptions =
    compilerGym.actions &&
    Object.keys(compilerGym.actions).map((x, i) => i + 1);

  useEffect(() => {
    const chartData =
      compilerGym.actions &&
      Object.entries(compilerGym.actions)
        .slice(0, actionSpace)
        .map(([name, action_id]) => ({
          name,
          action_id: action_id.toString(),
          children: [],
        }));
    setNodeSize({ x: actionSpace > 70 ? 700 : 300, y: 20 });
    setTreeData({ name: "root", action_id: "x", children: chartData });
    setTreeGlobalData({ name: "root", action_id: "x", children: chartData });

    return () => {};
  }, [actionSpace, compilerGym.actions]);

  /**
   * Recursive function to create new node + layer when user clicks on a node.
   *
   * @param {Array} arr Children's node.
   * @param {String} actionID Receives the tree action_id of the node.
   * @returns
   */
  const createNode = (arr, actionID) => {
    if (arr !== undefined) {
      arr.forEach((i) => {
        if (i.action_id === actionID) {
          i.children = treeData.children.map((o) => {
            return {
              name: o.name,
              action_id: `${o.action_id}.${layer}`,
              children: [],
            };
          });
        } else {
          createNode(i.children, actionID);
        }
      });
      return { name: "root", action_id: "x", children: arr };
    }
    return;
  };

  /**
   * Recursive function to delete a node in nested child object.
   *
   * @param {Array} arr Chiuldren array
   * @param {String} actionID Receives the tree action_id of the node.
   * @returns
   */
  const deleteNode = (arr, actionID) => {
    if (arr !== undefined) {
      arr.forEach((i) => {
        if (i.action_id === actionID) {
          i.children = [];
        } else {
          deleteNode(i.children, actionID);
        }
      });
      return { name: "root", action_id: "x", children: arr };
    }
    return;
  };

  const undoStep = () => {
    api.undoStep(session.session_id).then(
      (result) => {
        setSession({ ...session, ...result });
      },
      (error) => {
        console.log(error);
      }
    );
  };

  const handleNodeClick = (nodeDatum) => {
    let nodeActionID = nodeDatum.action_id.split(".")[0];
    let nodeDepth = nodeDatum.__rd3t.depth;

    if (nodeDatum !== undefined && nodeDepth !== 0) {
      if (nodeDepth === layer) {
        submitStep(nodeActionID);
        setLayer(layer + 1);
        setTreeGlobalData(
          createNode(treeGlobalData.children, nodeDatum.action_id)
        );
        setActionsTaken([...actionsTaken, nodeDatum.action_id]);
      } else if (
        nodeDepth === layer - 1 &&
        actionsTaken.includes(nodeDatum.action_id)
      ) {
        undoStep();
        setLayer(layer - 1);
        setTreeGlobalData(
          deleteNode(treeGlobalData.children, nodeDatum.action_id)
        );
        setActionsTaken((prev) =>
          prev.filter((i) => i !== nodeDatum.action_id)
        );
      }
      return;
    }
    return;
  };

  return (
    <>
      <div className="mx-2 searchtree-nav-wrapper">
        <Form.Row className="align-items-center">
          <Col md={4}>
            <InputGroup className="mb-1">
              <Dropdown
                as={InputGroup.Prepend}
                onSelect={(e) => setActionSpace(e)}
              >
                <Dropdown.Toggle variant="dark" id="dropdown-action-space">
                  Action Space
                </Dropdown.Toggle>
                <Dropdown.Menu as={DropdownMenu}>
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
        </Form.Row>
      </div>
      <div
        ref={treeWindow}
        className={classnames(
          "search-tree-container",
          { "dark-mode-tree": darkTheme },
          { "light-mode-tree": darkTheme === false }
        )}
      >
        <Tree
          data={treeGlobalData}
          nodeSize={nodeSize}
          translate={{ x: 10, y: treeWindow.current?.clientHeight / 3 || 10 }}
          renderCustomNodeElement={(rd3tProps) =>
            renderSvgNode({ ...rd3tProps, handleNodeClick })
          }
        />
      </div>
    </>
  );
};

export default ActionsSearchTree;

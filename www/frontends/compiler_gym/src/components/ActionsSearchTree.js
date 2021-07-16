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
import clone from "clone";
import ApiContext from "../context/ApiContext";
import ThemeContext from "../context/ThemeContext";
import Tree from "react-tree-graph";
import "react-tree-graph/dist/style.css";

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
              !value || child.props.children.toLowerCase().startsWith(value)
          )}
        </ul>
      </div>
    );
  }
);

const ActionsSearchTree = () => {
  const { compilerGym, submitStep } = useContext(ApiContext);
  const { darkTheme } = useContext(ThemeContext);
  const [actionSpace, setActionSpace] = useState(15);
  const [treeData, setTreeData] = useState({});
  const [activeNode, setActiveNode] = useState(null);
  const [layer, setLayer] = useState(0);

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
    setTreeData({ name: "root", action_id: "x", children: chartData });

    return () => {};
  }, [actionSpace, compilerGym.actions]);

  /**
   * Function to generate new nodes/layer when user clicks on a node
   *
   * @param {Object} layerData Receives a nested object data structure.
   * @returns
   */
  const getRoot = (layerData) => {
    if (layerData.action_id.split(".")[0] === activeNode) {
      return layerData;
    }
    for (let i = 0; i < layerData.children.length; i++) {
      let childJson = getRoot(layerData.children[i]);
      if (childJson) {
        let newBranch = {
          ...childJson,
          children: treeData.children.map((i) => {
            return {
              name: i.name,
              action_id: `${i.action_id}.${layer}`,
              children: [],
            };
          }),
        };
        return newBranch;
      }
    }
    return false;
  };

  const onNodeClick = (e, nodeKey) => {
    console.log(nodeKey);
    let nodeID = nodeKey.toString().split(".")[0];
    if (nodeKey !== undefined) {
      setActiveNode(nodeID);
      submitStep(nodeID);
      setLayer(layer + 1);
    }
  };

  let root = activeNode ? getRoot(treeData) : treeData;
  root = clone(root);

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
                <Dropdown.Menu as={CustomMenu}>
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
          { "": darkTheme === false }
        )}
      >
        <Tree
          data={root}
          height={2000}
          width={(treeWindow.current && treeWindow.current.clientWidth) || 400}
          animated
          keyProp={"action_id"}
          labelProp={"name"}
          gProps={{ onClick: onNodeClick }}
        />
      </div>
    </>
  );
};

export default ActionsSearchTree;

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext, useEffect, useRef } from "react";
import classnames from "classnames";
import ThemeContext from "../../context/ThemeContext";
import Tree from "react-d3-tree";
import { getYDeltas } from "../../utils/Helpers";

/**
 * Renders a node component.
 *
 * @param {Object} nodeDatum Receieves an object with node data.
 * @param {Function} handleNodeClick Function to be called when a node is clicked.
 * @param {Number} layer Takes the current active layer.
 * @returns {JSX} JSX element representing a node.
 */

const RenderSvgNode = ({
  nodeDatum,
  handleNodeClick,
  handleMouseOverTree,
  handleMouseOutTree,
  layer,
}) => {
  const foreignObjectProps = { width: 150, height: 200, x: 20, y: 10 };
  const nodeRef = useRef();
  const [seeDescription, setSeeDescription] = useState(false)

  return (
    <g ref={nodeRef}>
      <circle
        r="5"
        x="0"
        fill={nodeDatum.active ? "#2dce89" : "white"}
        strokeWidth="1"
        onClick={() => handleNodeClick(nodeDatum)}
        onMouseOver={() => {
          handleMouseOverTree(nodeDatum);
          setSeeDescription(true);
        }}
        onMouseOut={() => {
          handleMouseOutTree(nodeDatum);
          setSeeDescription(false);
        }}
      />
      <text
        strokeWidth={nodeDatum.active ? "1" : "0"}
        x="8"
        dy="4"
        onClick={() => handleNodeClick(nodeDatum)}
        onMouseOver={() => {
          handleMouseOverTree(nodeDatum);
          setSeeDescription(true);
        }}
        onMouseOut={() => {
          handleMouseOutTree(nodeDatum);
          setSeeDescription(false);
        }}
      >
        {nodeDatum.name}
      </text>

      {seeDescription && (
        <foreignObject {...foreignObjectProps}>
          <div className="active-node-info">
            <span className="description-tooltip2">
             {nodeDatum.description}
            </span>
          </div>
        </foreignObject>
      )}
    </g>
  );
};

/**
 * Renders the search tree component.
 *
 * @param {*} actionSpace Receives the number of nodes to show on an active node.
 * @param {Object} treeData Takes a hierarchical object data structure representeing the data of the tree.
 * @param {String} layer Receives the depth of the active node.
 * @param {Number} focusedNode a number representing the depth/level to position the tree when user clicks on a data point.
 * @param {Function} handleNodeClick Function that manages the logic of clicking on a node.
 * @returns
 */
const SearchTree = ({
  actionSpace,
  treeData,
  layer,
  focusedNode,
  handleNodeClick,
  handleMouseOverTree,
  handleMouseOutTree,
}) => {
  const { darkTheme } = useContext(ThemeContext);
  const treeWindow = useRef();
  const treeBox = useRef();

  const [nodeSize, setNodeSize] = useState({ x: 300, y: 20 });
  const [position, setPosition] = useState({
    x: 10,
    y: treeWindow.current?.clientHeight / 2.1 || 10,
  });
  const [deltas_y, setDeltas_y] = useState([]);

  useEffect(() => {
    const timer = setTimeout(() => {
      setDeltas_y(
        getYDeltas(
          treeBox.current?._reactInternals.child.lastEffect.stateNode.props
            .hierarchyPointNode
        ).reverse()
      );
    }, 500);
    return () => clearTimeout(timer);
  }, [treeData]);

  useEffect(() => {
    setNodeSize({ x: actionSpace > 100 ? 500 : 300, y: 20 });
    setPosition({
      x: focusedNode === 0 ? 10 : (focusedNode - 1) * -300,
      y:
        focusedNode === 0
          ? 300
          : treeWindow.current?.clientHeight / 2 -
            parseFloat(deltas_y[focusedNode]),
    });
    return () => {};
  }, [actionSpace, focusedNode, deltas_y]);

  return (
    <>
      <div
        ref={treeWindow}
        className={classnames(
          "search-tree-container",
          { "dark-mode-tree": darkTheme },
          { "light-mode-tree": darkTheme === false }
        )}
      >
        <Tree
          ref={treeBox}
          data={treeData}
          nodeSize={nodeSize}
          translate={position}
          scaleExtent={{ max: 1, min: 0.25 }}
          zoomable={true}
          renderCustomNodeElement={(rd3tProps) => (
            <RenderSvgNode
              {...rd3tProps}
              handleNodeClick={handleNodeClick}
              handleMouseOverTree={handleMouseOverTree}
              handleMouseOutTree={handleMouseOutTree}
              layer={layer}
            />
          )}
        />
      </div>
    </>
  );
};

export default SearchTree;

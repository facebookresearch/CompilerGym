/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext, useEffect, useRef } from "react";
import classnames from "classnames";
import ThemeContext from "../../context/ThemeContext";
import Tree from "react-d3-tree";

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
  const foreignObjectProps = { width: 115, height: 200, x: 20, y: 10 };

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
        dy={nodeDatum.active ? "-4" : "4"}
        onClick={() => handleNodeClick(nodeDatum)}
      >
        {nodeDatum.name}
      </text>
    </g>
  );
};

/**
 * Renders the search tree component.
 *
 * @param {*} actionSpace Receives the number of nodes to show on an active node.
 * @param {Object} treeData Takes a hierarchical object data structure representeing the data of the tree.
 * @param {String} layer Receives the depth of the active node.
 * @param {Function} handleNodeClick Function that manages the logic of clicking on a node.
 * @returns
 */
const SearchTree = ({ actionSpace, treeData, layer, handleNodeClick }) => {
  const { darkTheme } = useContext(ThemeContext);
  const treeWindow = useRef();

  const [nodeSize, setNodeSize] = useState({ x: 300, y: 20 });

  useEffect(() => {
    setNodeSize({ x: actionSpace > 100 ? 500 : 300, y: 20 });
    return () => {};
  }, [actionSpace]);

  const getDynamicPathClass = ({ source, target }) => {
    if (!target.children && target.depth < layer) return "link__to-leaf";
  };

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
          data={treeData}
          nodeSize={nodeSize}
          translate={{ x: 10, y: treeWindow.current?.clientHeight / 2 || 10 }}
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

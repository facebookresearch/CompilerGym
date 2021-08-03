/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

function isEmpty(obj) {
  return Object.keys(obj).length === 0;
}

/**
 * A function to build a hierarchical data structure object based on a flat array of action ids.
 *
 * @param {Array} ids represents the ids of actions with an specific order.
 * @param {Array} children a constant children array of objects containing data for each action.
 * @param {Array} rewards takes an array with the reward value on each step.
 * @returns an object with hierarchical structure to be display in a seacrh tree.
 */

export const makeTreeDataFromURL = (ids, children, rewards) => {
  if (ids.length && children !== undefined) {
    let actionsData = ids.map((i) => children.find((e) => e.action_id === i));
    let nestedChildren = actionsData?.reduceRight(
      (value, key, i) => ({
        name: key.name,
        action_id: `${key.action_id}.${i + 1}`,
        active: true,
        reward: rewards.slice(1, rewards.length)[i], // All rewards except the first value.
        children: isEmpty({ ...value })
          ? children.slice(0, 30).map((o) => {
              return {
                name: o.name,
                action_id: `${o.action_id}.${i + 2}`,
                children: [],
              };
            })
          : [{ ...value }],
      }),
      0
    );
    const treeData = {
      name: "root",
      action_id: "x",
      children: [nestedChildren],
    };
    return treeData;
  }
  return {
    name: "root",
    action_id: "x",
    children: [{ name: "tempRoot", action_id: "x", children: [] }],
  };
};

/**
 * Takes a string and convert it into an array of actions.
 *
 * @param {String} commandLine the input as command line.
 * @param {Array} actionsList array of objects containing the name and id of each action.
 * @returns
 */
export const getCommandLineArray = (commandLine, actionsList) => {
  const clArray = commandLine.split(" input.bc -o output.bc")[0].split(" ");
  let actionNames = clArray.slice(1, clArray.length); // All actions except the 'opt' command.
  let temp = [];
  actionNames.forEach((i) => {
    actionsList.forEach((o) => {
      if (o.name === i) {
        temp.push(o.action_id);
      }
    });
  });
  return temp;
};

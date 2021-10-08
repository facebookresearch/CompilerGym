/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * An utility function to test for an empty object.
 * @param {Object} obj
 * @returns {Boolean}
 */
const isEmpty = (obj) => {
  return Object.keys(obj).length === 0;
};

/**
 * Function to filter an array of objects with same name.
 * @param {Array} list an array of objects.
 * @returns
 */
const getUniqueValues = (list) => {
  return [...new Map(list.map((item) => [item.name, item])).values()];
};

/**
 * A function to build a hierarchical data structure object based on a flat array session states.
 * It builds an nested object.
 *
 * @param {Array} states represents array of states in current CompilerGym session.
 * @param {Array} children a constant children array of objects containing data for each action.
 * @param {String} commandLine a string representing the commandline input.
 * @returns an object with hierarchical structure to be display in a seacrh tree.
 */
export const makeSessionTreeData = (states, children, commandLine) => {
  if (states.length >= 2 ) {
    const ids = getCommandLineArray(commandLine, children)
    const rewards = states.map((a) => a.reward).slice(1, states.length); // All rewards except the initial 0.

    let actionsData = ids
      .map((i) => children.find((e) => e.action_id === i))
      .map((i, index) => ({
        ...i,
        children: children
          .slice(0, 30)
          .map((o) => ({ ...o, action_id: `${o.action_id}.${index + 2}` })),
      })); // Adds children to parents to display as options in the tree.

    let nestedChildren = actionsData?.reduceRight(
      (value, current, i) => ({
        name: current.name,
        description: current.description,
        action_id: `${current.action_id}.${i + 1}`,
        active: true,
        reward: rewards[i].toFixed(3),
        children: isEmpty({ ...value })
          ? children.slice(0, 30).map((o) => {
              return {
                name: o.name,
                description: o.description,
                action_id: `${o.action_id}.${i + 2}`,
                children: [],
              };
            })
          : getUniqueValues([...current.children, { ...value }]),
      }),
      {}
    );

    const treeData = {
      name: "root",
      action_id: "x",
      children: getUniqueValues([...children.slice(0, 30), nestedChildren]),
    };
    return treeData;
  }
  return {
    name: "root",
    action_id: "x",
    children: children.slice(0, 30),
  };
};

/**
 * Takes a string representing a command line and convert it into an array of actionsIds.
 *
 * @param {String} commandLine the input as command line.
 * @param {Array} actionsList an array of objects containing the name and id of each CompilerGym action.
 * @returns
 */
export const getCommandLineArray = (commandLine, actionsList) => {
  if (commandLine !== undefined) {
    const clArray = commandLine.split(" input.bc -o output.bc")[0].split(" ");
    let actionNames = clArray.slice(1, clArray.length); // All actions except the 'opt' command string.
    let temp = [];
    actionNames.forEach((i) => {
      actionsList.forEach((o) => {
        if (o.name === i) {
          temp.push(o.action_id);
        }
      });
    });
    return temp;
  }
  return;
};

/**
 * Group objects by key
 * @param {*} objectArray
 * @param {*} property
 * @returns
 */
export const groupBy = (objectArray, property) => {
  return objectArray.reduce((acc, obj) => {
    let key = property;
    if (!acc[key]) {
      acc[key] = [];
    }
    acc[key].push(obj[property]);
    return acc;
  }, {});
};

/**
 * Returns the positive or negative greatest absoulute value with sign.
 * @param {Array} array receives an array of positive and negative numbers.
 * @returns
 */
export const getMaxDelta = (array) => {
  let max = Math.max(...array);
  let min = Math.min(...array);
  if (Math.abs(max) > Math.abs(min)) return max;
  return min;
};

/**
 * Calculates the percentage increase between two numbers with edge case scenarios.
 * @param {Number} a first number.
 * @param {Number} b second number.
 * @returns
 */
export const percIncrease = (a, b) => {
  let percent;
  if (b !== 0) {
    if (a !== 0) {
      percent = ((b - a) / a) * 100;
    } else {
      percent = b;
    }
  } else {
    percent = -a * 100;
  }
  return Math.floor(percent);
};

/**
 * This recursive function takes a tree object of parents and returns the translate Y position in canvas as a flat array.
 *
 * @param {*} obj the three object of parents
 * @param {*} y_deltas a memoized array with y deltas.
 * @returns
 */
export const getYDeltas = (obj, y_deltas = []) => {
  obj &&
    Object.entries(obj).forEach(([key, value]) => {
      if (key === "parent" && value) {
        y_deltas.push(value.x);
        return getYDeltas(value, y_deltas);
      } else if (key === "parent" && !value) {
        return y_deltas;
      }
    });

  return y_deltas;
};

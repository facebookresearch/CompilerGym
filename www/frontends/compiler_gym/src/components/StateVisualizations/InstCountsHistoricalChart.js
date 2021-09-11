/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useEffect, Fragment } from "react";
import classnames from "classnames";
import { groupBy, getMaxDelta, percIncrease } from "../../utils/Helpers";
import InstCountDict from "../../utils/InstCountDict";
import SparkLineTable from "./SparkLineTable";

const InstCountsHistoricalChart = ({ sessionStates, commandLine, darkTheme, sortBy }) => {
  const [steps, setSteps] = useState([]);

  useEffect(() => {
    setSteps(commandLine.split(" input.bc -o output.bc")[0].split(" "));
    return () => {};
  }, [commandLine]);

  const sortedList = (array, name) => {
    switch (name) {
      case "result":
        return array.sort((a, b) => (a["maxValue"] > b["maxValue"] ? -1 : 1));
      case "diff":
        return array.sort((a, b) => (a["maxDelta"] > b["maxDelta"] ? -1 : 1));
      default:
        return array;
    }
  };

  const generateHistoricalData = () => {
    let groupedItems = [];
    const historicalObservations = sessionStates.map((i) => i.instcount);
    const keys = Object.keys(historicalObservations[0]);
    keys.forEach((name) => {
      let observation = groupBy(historicalObservations, name);
      groupedItems.push(observation);
    });
    const output = groupedItems
      .map((i) =>
        Object.entries(i).map(([category, result]) => ({
          category,
          result,
          percentageChange: result.map((v, i) =>
            i === 0 ? 0 : percIncrease(result[i - 1], v)
          ),
          maxValue: Math.max(...result),
          maxDelta: getMaxDelta(
            result.map((v, i) => (i === 0 ? 0 : percIncrease(result[i - 1], v)))
          ),
        }))
      )
      .flat();
    return output;
  };

  const tableData = sortedList(generateHistoricalData(), sortBy);

  const rows = tableData.map((i, index) => {
    let description = InstCountDict.find((o) => `${o.Name}Count` === i.category)?.Description ?? "";
    let sparklineData = `${i.result.join(", ")}; ${steps.join(", ")}`;
    let sparklineDeltaData = `${i.percentageChange.join(", ")}; ${steps.join(
      ", "
    )}`;
    return (
      <Fragment key={index}>
        <tr>
          <td className="cell-info">
            {i.category}{" "}
            <span className="description-tooltip">{description}</span>
          </td>
          <td className="text-right">{i.maxValue}</td>
          <td data-sparkline={sparklineData} />
          <td className="text-right">{i.maxDelta} %</td>
          <td data-sparkline={sparklineDeltaData} />
        </tr>
      </Fragment>
    );
  });

  return (
    <div
      className={classnames(
        "mt-2 pt-2",
        { "bg-dark text-white": darkTheme },
        { "": !darkTheme }
      )}
    >
      <h4 className="text-center">Historical Values</h4>
      <SparkLineTable>
        <thead>
          <tr>
            <th>InstCounts</th>
            <th className="text-tight">Max Value</th>
            <th className="text-center">Trend</th>
            <th className="text-tight">Max Change</th>
            <th className="text-center">Percentage Change</th>
          </tr>
        </thead>
        <tbody id="tbody-sparkline">{rows}</tbody>
      </SparkLineTable>
    </div>
  );
};

export default InstCountsHistoricalChart;

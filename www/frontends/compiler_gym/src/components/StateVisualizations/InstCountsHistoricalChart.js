import React, { useState, useEffect, Fragment } from "react";
import classnames from "classnames";
import { groupBy, getMaxDelta, percIncrease } from "../../utils/Helpers";
import SparkLineTable from "./SparkLineTable";

const InstCountsHistoricalChart = ({ sessionStates, darkTheme, sortBy }) => {
  const [steps, setSteps] = useState([]);

  useEffect(() => {
    let lastState = sessionStates?.[sessionStates?.length - 1];
    setSteps(
      lastState?.commandline.split(" input.bc -o output.bc")[0].split(" ")
    );
    return () => {};
  }, [sessionStates]);

  const sortedList = (array, name) => {
    switch (name) {
      case "result":
        return array.sort((a, b) => (a["maxValue"] > b["maxValue"] ? -1 : 1));
      case "diff":
        return array.sort((a, b) => (a["maxDelta"] > b["maxDelta"] ? -1 : 1));
      case "category":
        return array.sort((a, b) => (a["caregory"] > b["category"] ? -1 : 1));
      default:
        return [];
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
    let sparklineData = `${i.result.join(", ")}; ${steps.join(", ")}`;
    let sparklineDeltaData = `${i.percentageChange.join(", ")}; ${steps.join(
      ", "
    )}`;
    return (
      <Fragment key={index}>
        <tr>
          <td>{i.category}</td>
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

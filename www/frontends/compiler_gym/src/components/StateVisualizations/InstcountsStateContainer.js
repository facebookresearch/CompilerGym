/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState } from "react";
import classnames from "classnames";
import { Dropdown, Row, Col } from "react-bootstrap";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import InstCountsHistoricalChart from "./InstCountsHistoricalChart";

const InstcountsStateContainer = ({
  sessionStates,
  instcount,
  prev_instcount,
  darkTheme,
}) => {
  const [sortBy, setSortBy] = useState("result");
  const [toggle, setToggle] = useState(true);

  const newData = Object.entries(instcount).map(([category, result]) => ({
    category,
    result,
  }));

  const prevChartData = Object.entries(prev_instcount).map(
    ([category, prevResult]) => ({
      category,
      prevResult,
    })
  );

  /**
   * Function to merge new observation data with previous observation data.
   *
   * @param {Array} a1 Takes current/new observation array.
   * @param {Array} a2 Takes previous observation state.
   * @returns an array containg both observation states
   */
  const mergeByKey = (a1, a2) =>
    a1.map((i) => ({
      ...a2.find((o) => o.category === i.category && o),
      ...i,
    }));

  const chartData = mergeByKey(newData, prevChartData)
    .map((i) => {
      return {
        category: i.category,
        result: i.result,
        diff: (i.result - i.prevResult) / i.result || 0,
      };
    })
    .sort((a, b) => (a[sortBy] > b[sortBy] ? -1 : 1));

  const options = {
    colors: ["#64E572", "#DDDF00"],
    chart: {
      height: 175 + "%", //3:2 ratio
    },
    title: {
      text: "InstCounts",
      style: {
        color: darkTheme && "white",
      },
    },
    xAxis: {
      categories: chartData.map((i) => i.category),
      labels: {
        style: {
          color: darkTheme && "white",
        },
      },
    },
    yAxis: [
      {
        // Primary yAxis
        title: {
          text: "Counts",
          style: {
            color: darkTheme && "white",
          },
        },
        labels: {
          style: {
            color: darkTheme && "white",
          },
        },
      },
      {
        // Secondary yAxis
        title: {
          text: "Delta",
          style: {
            color: darkTheme && "white",
          },
        },
        labels: {
          format: "{value}°%",
        },
        opposite: true,
      },
    ],
    tooltip: {
      shared: true,
    },
    legend: {
      enabled: false,
    },
    plotOptions: {
      series: {
        pointPadding: 0.4,
        borderWidth: 0,
        dataLabels: {
          format: "{point.y}",
          borderWidth: 0,
          color: darkTheme && "white",
          style: {
            fontFamily: "menlo",
          },
        },
      },
    },
    credits: {
      enabled: false,
    },
    series: [
      {
        name: "InstCounts",
        type: "bar",
        data: chartData.map((i) => i.result),
        dataLabels: {
          enabled: true,
        },
      },
      {
        name: "Delta",
        type: "bar",
        data: chartData.map((i) => Math.round(i.diff * 100 * 100) / 100),
        tooltip: {
          valueSuffix: "%",
        },
        yAxis: 1,
      },
    ],
    exporting: {
      buttons: {
        contextButton: {
          menuItems: ["viewFullscreen", "printChart"],
        },
      },
    },
  };

  return (
    <div
      className={classnames(
        "chart-container",
        { "chart-dark-mode bg-dark text-white": darkTheme },
      )}
    >
      <Row className="align-items-center mx-0 pt-2">
        <Col md="3"></Col>
        <Col md="9" className="text-right">
          <div className="px-2">
            <h5>
              The InstCount observation space is a 70-dimension integer feature
              vector in the range [0,∞]. It contains the total
              number of instructions, the total number of basic blocks, and the
              total number of functions. The remaining features are the number
              of instructions of each of the 67 different types in the program.
              <br />
              <a
                className={classnames({ "text-white": darkTheme })}
                href="https://facebookresearch.github.io/CompilerGym/llvm/index.html#instcount"
                target="_blank"
                rel="noreferrer"
              >
                Docs <i className="bi bi-link-45deg" />
              </a>
            </h5>
          </div>
        </Col>
      </Row>
      <div className="row-sort-button ml-2">
        <Row className="align-items-center">
          <Col>
            <Dropdown onSelect={(e) => setSortBy(e)}>
              <Dropdown.Toggle id="sort-options" size="sm">
                Sort
              </Dropdown.Toggle>

              <Dropdown.Menu>
                <Dropdown.Item
                  eventKey={"result"}
                  active={sortBy === "result" ? true : false}
                >
                  Values
                </Dropdown.Item>
                <Dropdown.Item
                  eventKey={"diff"}
                  active={sortBy === "diff" ? true : false}
                >
                  Delta
                </Dropdown.Item>
                <Dropdown.Item
                  eventKey={"category"}
                  active={sortBy === "category" ? true : false}
                >
                  Alphabetical
                </Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown>
          </Col>
          <Col className="px-0">
            <div className="switch_box box_1">
              <input
                type="checkbox"
                className="switch_1"
                checked={toggle}
                onChange={() => setToggle(!toggle)}
              />
            </div>
          </Col>
        </Row>
      </div>
      {toggle ? (
        <InstCountsHistoricalChart
          sessionStates={sessionStates}
          darkTheme={darkTheme}
          sortBy={sortBy}
        />
      ) : (
        <HighchartsReact highcharts={Highcharts} options={options} />
      )}
    </div>
  );
};

export default InstcountsStateContainer;

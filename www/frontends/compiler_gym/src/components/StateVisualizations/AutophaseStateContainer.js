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
import HC_exporting from "highcharts/modules/exporting";
import HC_exportData from "highcharts/modules/export-data";
import AutophaseHistoricalChart from "./AutophaseHistoricalChart";

HC_exporting(Highcharts);
HC_exportData(Highcharts);

const AutophaseStateContainer = ({
  sessionStates,
  commandLine,
  autophase,
  prev_authophase,
  darkTheme,
}) => {
  const [sortBy, setSortBy] = useState("result");
  const [toggle, setToggle] = useState(true);

  const newData = Object.entries(autophase).map(([category, result]) => ({
    category,
    result,
  }));

  const prevChartData = Object.entries(prev_authophase).map(
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
    colors: ["#7cb5ec", "#DDDF00"],
    chart: {
      type: "bar",
      height: 180 + "%",
    },
    title: {
      text: "AutoPhase [Last Action]",
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
          color: darkTheme && "white",
        },
      },
    },
    credits: {
      enabled: false,
    },

    series: [
      {
        name: "Autophase",
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
        <Col md="6">
          <div className="px-2">
            <h5>
              The Autophase observation space is a 56-dimension integer feature
              vector summarizing the static LLVM-IR representation.
              <br />
              <a
                className={classnames({ "text-white": darkTheme })}
                href="https://facebookresearch.github.io/CompilerGym/llvm/index.html#autophase"
                target="_blank"
                rel="noreferrer"
              >
                Docs <i className="bi bi-link-45deg" />
              </a>
            </h5>
          </div>
        </Col>
        <Col md="6"></Col>
      </Row>
      <div className="row-sort-button mr-3">
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
                  Value
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
        <AutophaseHistoricalChart
          sessionStates={sessionStates}
          commandLine={commandLine}
          darkTheme={darkTheme}
          sortBy={sortBy}
        />
      ) : (
        <HighchartsReact highcharts={Highcharts} options={options} />
      )}
    </div>
  );
};

export default AutophaseStateContainer;

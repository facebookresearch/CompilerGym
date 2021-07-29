/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext, useEffect } from "react";
import classnames from "classnames";
import { Offcanvas } from "react-bootstrap";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import ThemeContext from "../../context/ThemeContext";

const RewardHistoryChart = ({ session, show, onHide }) => {
  const { darkTheme } = useContext(ThemeContext);
  const [rewards, setRewards] = useState([]);
  const [cumulativeSum, setCumulativeSum] = useState([]);
  const [steps, setSteps] = useState([]);

  useEffect(() => {
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));
    let lastState = session.states?.[session.states?.length - 1];
    setRewards(rewards);
    setCumulativeSum(
      rewards?.reduce((a, x, i) => [...a, a.length > 0 ? x + a[i - 1] : x], [])
    );
    setSteps(
      lastState?.commandline.split(" input.bc -o output.bc")[0].split(" ")
    );
    return () => {};
  }, [session]);

  const options = {
    colors: darkTheme ? ["#b34949", "#2593B8"] : ["#2593B8", "#434348"],
    chart: {
      type: "areaspline",
    },
    title: {
      text: "Reward History",
      style: {
        color: darkTheme && "white",
      },
    },
    xAxis: {
      categories: steps,
      labels: {
        style: {
          color: darkTheme && "white",
        },
      },
    },
    yAxis: {
      title: {
        text: "Reward Value",
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
    credits: {
      enabled: false,
    },
    legend: {
      backgroundColor: "white",
    },
    plotOptions: {
      series: {
          pointWidth: 20
      }
    },
    series: [
      {
        name: "Cumulative Reward",
        data: cumulativeSum.map(i => parseFloat(i.toFixed(3))),
      },
      {
        type: 'column',
        name: 'Reward',
        data: rewards,
      },
    ],
    exporting: {
      buttons: {
        contextButton: {
          menuItems: ["viewFullscreen", "printChart", "downloadCSV"],
        },
      },
    },
  };

  return (
    <>
      <Offcanvas show={show} onHide={onHide} placement={"bottom"}>
        <Offcanvas.Header
          closeButton
          className={classnames(
            { "offcanvas-dark-mode": darkTheme },
            { "": darkTheme === false }
          )}
        >
          <Offcanvas.Title>Reward History</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body
          className={classnames(
            { "offcanvas-dark-mode chart-dark-mode": darkTheme },
            { "": darkTheme === false }
          )}
        >
          <HighchartsReact highcharts={Highcharts} options={options} />
        </Offcanvas.Body>
      </Offcanvas>
    </>
  );
};

export default RewardHistoryChart;

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useContext, useEffect, useRef } from "react";
import classnames from "classnames";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import ThemeContext from "../../context/ThemeContext";

const RewardHistoryChart = ({ session }) => {
  const { darkTheme } = useContext(ThemeContext);
  const panel = useRef();
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
      height: 18 + "%",
      width: panel.current?.clientWidth,
    },
    title: null,
    xAxis: {
      categories: steps,
      labels: {
        enabled: false,
      },
    },
    yAxis: {
      title: null,
      labels: {
        enabled: true,
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
        pointWidth: 5,
        allowPointSelect: true,
        states: {
          hover: {
            enabled: false,
          },
          select: {
            color: "yelow",
          },
          inactive: {
            enabled: false,
            opacity: 1,
          },
        },
      },
    },
    series: [
      {
        type: "line",
        name: "Cumulative Reward",
        data: cumulativeSum.map((i) => parseFloat(i.toFixed(3))),
      },
      {
        type: "column",
        name: "Reward",
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
    <div
      ref={panel}
      className={classnames(
        { "offcanvas-dark-mode chart-dark-mode": darkTheme },
        { "": !darkTheme }
      )}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  );
};

export default RewardHistoryChart;

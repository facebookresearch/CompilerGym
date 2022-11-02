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

const RewardHistoryChart = ({ session, highlightedPoint, handleClickOnChart }) => {
  const { darkTheme } = useContext(ThemeContext);
  const panel = useRef();
  const chart = useRef();
  const [rewards, setRewards] = useState([]);
  const [cumulativeSum, setCumulativeSum] = useState([]);
  const [steps, setSteps] = useState([]);

  useEffect(() => {
    let rewards = session.states?.map((i) => parseFloat(i.reward.toFixed(3)));
    let lastState = session?.commandline
    setRewards(rewards);
    setCumulativeSum(
      rewards?.reduce((a, x, i) => [...a, a.length > 0 ? x + a[i - 1] : x], [])
    );
    setSteps( lastState?.split(" input.bc -o output.bc")[0].split(" "));
    return () => {};
  }, [session]);

  useEffect(() => {
    if (highlightedPoint && highlightedPoint.selected) {
      chart.current?.chart.tooltip.refresh(chart.current?.chart.series[1].points[highlightedPoint.point]);
    } else {
      chart.current?.chart.tooltip.hide();
    }
    return () => {};
  }, [highlightedPoint]);

  const setClickPoint = (e) => {
    handleClickOnChart(e);
  };

  const options = {
    colors: darkTheme ? ["#b34949", "#2593B8"] : ["#2593B8", "#434348"],
    chart: {
      type: "areaspline",
      height: 18 + "%",
      width: panel.current?.clientWidth,
      animation: false,
      style: { fontFamily: "'Menlo', sans-serif" },
    },
    title: {
      text: "Rewards",
      floating: true,
      align: "left",
      x: panel.current?.clientWidth / 2,
      y: -1,
      style: {
        fontSize: "12px",
        color: darkTheme && "white",
      },
    },
    xAxis: {
      categories: steps,
      labels: {
        enabled: false,
      },
      crosshair: true
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
      align: "right",
      verticalAlign: "top",
      layout: "vertical",
      backgroundColor: "white",
      x: -10,
      y: 50,
      floating: true,
      itemStyle: {
        fontWeight: "normal",
        fontSize: "11px",
      },
    },
    tooltip: {
      shared: true,
    },
    plotOptions: {
      series: {
        pointWidth: 4,
        marker: {
          radius: 2,
        },
        allowPointSelect: true,
        point: {
          events: {
            select: function () {
              setClickPoint(this);
            },
          },
        },
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
        name: "Cumulative",
        data: cumulativeSum.map((i) => parseFloat(i.toFixed(3))),
      },
      {
        type: "column",
        name: "Step Reward",
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
      <HighchartsReact ref={chart} highcharts={Highcharts} options={options} />
    </div>
  );
};

export default RewardHistoryChart;

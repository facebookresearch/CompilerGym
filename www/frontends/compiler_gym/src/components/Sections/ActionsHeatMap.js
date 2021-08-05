/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect, useState, useRef, useContext } from "react";
import classnames from "classnames";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import highchartsHeatmap from "highcharts/modules/heatmap";
import ThemeContext from "../../context/ThemeContext";

highchartsHeatmap(Highcharts);

const ActionsHeatMap = ({ session, highlightedPoint }) => {
  const heatMapWindow = useRef();
  const chart = useRef();
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

  useEffect(() => {
    if (highlightedPoint && highlightedPoint.selected) {
      chart.current?.chart.series[0].points[highlightedPoint.point]?.select();
    } else {
      chart.current?.chart.series[0].points[highlightedPoint.point]?.select(
        false
      );
    }
    return () => {};
  }, [highlightedPoint]);

  const getPointCategoryName = (point, dimension) => {
    var series = point.series,
      isY = dimension === "y",
      axis = series[isY ? "yAxis" : "xAxis"];
    return axis.categories[point[isY ? "y" : "x"]];
  };

  const setHoverData = (e) => {
    console.log(e);
  };

  const options = {
    chart: {
      type: "heatmap",
      plotBorderWidth: 0,
      width: heatMapWindow.current?.clientWidth,
      height: 55,
      marginTop: 10,
    },

    title: null,

    xAxis: {
      categories: steps,
      title: null,
      labels: {
        enabled: false,
      },
    },

    yAxis: {
      categories: ["Reward", "CumulativeReward"],
      title: null,
      reversed: true,
      labels: {
        style: {
          color: darkTheme && "white",
        },
      },
    },

    accessibility: {
      point: {
        descriptionFormatter: function (point) {
          var ix = point.index + 1,
            xName = getPointCategoryName(point, "x"),
            yName = getPointCategoryName(point, "y"),
            val = point.value;
          return ix + ". " + xName + " sales " + yName + ", " + val + ".";
        },
      },
    },

    colorAxis: {
      min: Math.min(...rewards),
      minColor: "#FFFFFF",
      maxColor: Highcharts.getOptions().colors[7],
    },

    tooltip: {
      formatter: function () {
        return (
          "<b>" +
          getPointCategoryName(this.point, "x") +
          "</b>  <br><b>" +
          getPointCategoryName(this.point, "y") +
          ": </b>" +
          this.point.value
        );
      },
    },

    credits: {
      enabled: false,
    },
    legend: {
      enabled: false,
    },
    plotOptions: {
      series: {
        allowPointSelect: true,
        point: {
          events: {
            mouseOver: function () {
              setHoverData(this);
            },
          },
        },
        states: {
          hover: {
            enabled: false,
          },
          select: {
            color: function () {
              return this.point.color;
            },
            opacity: 0.5,
            borderWidth: 2,
            lineWidth: 1.5,

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
        name: "Reward",
        borderWidth: 0,
        data: rewards.map((i, index) => [index, 0, i]),
      },
      {
        name: "CumulativeReward",
        borderWidth: 0,
        data: cumulativeSum.map((i, index) => [index, 1, i]),
      },
    ],

    exporting: {
      buttons: {
        contextButton: {
          enabled: false,
          menuItems: ["viewFullscreen", "printChart", "downloadCSV"],
        },
      },
    },
  };

  return (
    <div
      ref={heatMapWindow}
      className={classnames(
        "highcharts-figure",
        { "chart-dark-mode": darkTheme },
        { "": !darkTheme }
      )}
    >
      <HighchartsReact ref={chart} highcharts={Highcharts} options={options} />
    </div>
  );
};

export default ActionsHeatMap;

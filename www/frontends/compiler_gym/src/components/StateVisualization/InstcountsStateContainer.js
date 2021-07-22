/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from "react";
import classnames from "classnames";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

const InstcountsStateContainer = ({ instcount, darkTheme }) => {

  const chartData = Object.entries(instcount).map(([category, result]) => ({
    category,
    result,
  }));

  const options = {
    colors: ["#64E572", "#DDDF00", "#FF9655", "#FFF263", "#6AF9C4"],
    chart: {
      type: "bar",
      height: 150 + "%", //3:2 ratio
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
    yAxis: {
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
    legend: {
      enabled: false,
    },
    plotOptions: {
      series: {
        pointPadding: 0.4,
        borderWidth: 0,
        dataLabels: {
          enabled: true,
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
        data: chartData.map((i) => i.result),
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
      className={classnames(
        "chart-container",
        { "chart-dark-mode": darkTheme },
        { "": darkTheme === false }
      )}
    >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  );
};

export default InstcountsStateContainer;

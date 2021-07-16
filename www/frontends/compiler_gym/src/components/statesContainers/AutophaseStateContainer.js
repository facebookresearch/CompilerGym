/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useContext } from "react";
import classnames from "classnames";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import ThemeContext from "../../context/ThemeContext";

const AutophaseStateContainer = ({ autophase }) => {
  const { darkTheme } = useContext(ThemeContext);

  const chartData = Object.entries(autophase).map(([category, result]) => ({
    category,
    result,
  }));

  const options = {
    chart: {
      type: "bar",
      height: 150 + "%", //3:2 ratio
    },
    title: {
      text: "AutoPhase",
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
        data: chartData.map((i) => i.result),
      },
    ],
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

export default AutophaseStateContainer;

import React, { useContext } from "react";
import classnames from "classnames";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import ThemeContext from "../../context/ThemeContext";

const InstcountsStateContainer = ({ instcount }) => {
  const { darkTheme } = useContext(ThemeContext);

  const options = {
    colors: ["#64E572", "#DDDF00", "#FF9655", "#FFF263", "#6AF9C4"],
    chart: {
      type: "bar",
    },
    title: {
      text: "InstCounts",
      style: {
        color: darkTheme && "white",
      },
    },
    xAxis: {
      categories: [
        "TotalInstCount",
        "TotalBlocksCount",
        "TotalFuncsCount",
        "RectCount",
        "BrCount",
      ],
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
        data: instcount,
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

export default InstcountsStateContainer;

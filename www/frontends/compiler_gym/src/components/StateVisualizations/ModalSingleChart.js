import React from "react";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";

const ModalSingleChart = ({ data, darkTheme }) => {
  const options = {
    colors: ["#2593B8", "#434348"],
    chart: {
      type: "area",
    },
    title: {
      text: undefined,
    },
    xAxis: {
      categories: data.xAxis.categories,
      labels: {
        enabled: true,
        style: {
          color: darkTheme && "white",
        },
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
      enabled: false,
    },
    tooltip: {
      formatter: function () {
        return (
          "Action: <b>" + this.x + "</b> <br/> Value: <b>" + this.y + "</b>"
        );
      },
    },
    plotOptions: {
      series: {
        pointWidth: 4,
        marker: {
          radius: 2,
        },
      },
    },
    series: [
      {
        data: data.series[0].data,
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

  return <HighchartsReact highcharts={Highcharts} options={options} />;
};

export default ModalSingleChart;

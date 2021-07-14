import React, { useContext } from "react";
import classnames from "classnames";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import ThemeContext from "../../context/ThemeContext";

const AutophaseStateContainer = ({ autophase }) => {
  const { darkTheme } = useContext(ThemeContext);

  const options = {
    chart: {
      type: "bar",
    },
    title: {
      text: "AutoPhase",
      style: {
        color: darkTheme && "white",
      },
    },
    xAxis: {
      categories: [
        "BBNumArgsHi",
        "BBNumArgsLo",
        "onePred",
        "OnePredOneSuc",
        "OnePredTwoSuc",
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
          color: darkTheme && "white",
        },
      },
    },
    credits: {
      enabled: false,
    },
    //tooltip: {
    //headerFormat: '<span style="font-size:11px">{series.name}</span><br>',
    //pointFormat: '<span style="color:{point.color}">{point.x}</span>: <b>{point.y}</b> <br/>'
    //},

    series: [
      {
        name: "Autophase",
        data: autophase,
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

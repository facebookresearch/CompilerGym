/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect, useRef } from "react";
import Highcharts from "highcharts";

const defaultOptions = {
  chart: {
    backgroundColor: null,
    borderWidth: 0,
    type: "area",
    margin: [2, 0, 2, 0],
    width: 160,
    height: 20,
    style: {
      overflow: "visible",
    },
    // small optimization, saves 1-2 ms each sparkline
    skipClone: true,
  },
  title: {
    text: "",
  },
  credits: {
    enabled: false,
  },
  xAxis: {
    labels: {
      enabled: false,
    },
    title: {
      text: null,
    },
    startOnTick: false,
    endOnTick: false,
    tickPositions: [],
  },
  yAxis: {
    endOnTick: false,
    startOnTick: false,
    labels: {
      enabled: false,
    },
    title: {
      text: null,
    },
    tickPositions: [0],
  },
  legend: {
    enabled: false,
  },
  tooltip: {
    backgroundColor: "white",
    borderWidth: 1,
    useHTML: true,
    hideDelay: 0,
    padding: 6,
    borderColor: "silver",
    borderRadius: 3,
    outside: true,
  },
  plotOptions: {
    series: {
      animation: false,
      lineWidth: 1,
      shadow: false,
      states: {
        hover: {
          lineWidth: 1,
        },
      },
      marker: {
        radius: 1,
        states: {
          hover: {
            radius: 2,
          },
        },
      },
      fillOpacity: 0.25,
    },
    column: {
      negativeColor: "#910000",
      borderColor: "silver",
    },
  },

  series: [
    {
      data: [1, 2, 3],
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

const SparkLine = (props) => {
  const container = useRef();

  useEffect(() => {
    const options = Highcharts.merge(defaultOptions, props.options);
    let chart = Highcharts.chart(container.current, options);
    return () => {
      chart.destroy();
    };
  }, [props.options]);

  const handleClick = (data) => {
    props.setModal(true)
    props.setModalData(data)
  }

  return <td ref={container} onClick={() => handleClick(props.options)}></td>;
};

class SparkLineTable extends React.Component {
  toSparkLine(children, params) {
    let header;

    return React.Children.map(children, (child) => {
      if (!React.isValidElement(child)) return child;

      if (child.type === "th") header = child.props.children;

      if (child.props["data-sparkline"]) {
        return this.sparkLine(child, header);
      }

      if (child.props.children) {
        child = React.cloneElement(child, {
          children: this.toSparkLine(child.props.children),
        });
      }

      return child;
    });
  }

  sparkLine(element, header) {
    const dataAttr = element.props["data-sparkline"].split("; ");
    const data = dataAttr[0].split(", ").map(Number);

    const options = {
      chart: {
        type: "area",
      },
      xAxis: {
        categories: dataAttr[1] && dataAttr[1].split(", "),
      },
      plotOptions: {
        series: {
          color: "#2593B8",
        },
      },
      series: [
        {
          data: data,
          pointStart: 0,
        },
      ],

      tooltip: {
        useHTML: true,
        headerFormat: `<span><b>{point.x}</b></span><br/>`,
        pointFormat: `<span><b>Value:</b> {point.y}</span>`,
      },
    };

    return (
      <SparkLine
        options={options}
        setModal={this.props.setModal}
        setModalData={this.props.setModalData}
      />
    );
  }

  render() {
    const style = {
      margin: "0 auto",
      borderCollapse: "collapse",
    };

    return (
      <table
        style={style}
        id="table-sparkline"
        className="sparkline-table mt-3"
      >
        {this.toSparkLine(this.props.children)}
      </table>
    );
  }
}

export default SparkLineTable;

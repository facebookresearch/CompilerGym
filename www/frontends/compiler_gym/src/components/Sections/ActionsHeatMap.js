/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, {useRef} from "react";
//import classnames from "classnames";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import highchartsHeatmap from "highcharts/modules/heatmap";
//import ThemeContext from "../../context/ThemeContext";

highchartsHeatmap(Highcharts);

const ActionsHeatMap = ({session}) => {
  const heatMapWindow = useRef();

  function getPointCategoryName(point, dimension) {
    var series = point.series,
        isY = dimension === 'y',
        axis = series[isY ? 'yAxis' : 'xAxis'];
    return axis.categories[point[isY ? 'y' : 'x']];
}

  const options = {
    chart: {
      type: 'heatmap',
      plotBorderWidth: 0.5,
      width: heatMapWindow.current?.clientWidth,
      height: 80,
      marginTop: 20
    },

    title: null,

  xAxis: {
      categories: ['-ss', '-sadsd', '-mom230', '-sdash', '-symplify', '-barrier', '-ee-instrument', ],
      title: null,
  },

  yAxis: {
      categories: ['Reward', 'maxReward'],
      title: null,
      reversed: true
  },

  accessibility: {
      point: {
          descriptionFormatter: function (point) {
              var ix = point.index + 1,
                  xName = getPointCategoryName(point, 'x'),
                  yName = getPointCategoryName(point, 'y'),
                  val = point.value;
              return ix + '. ' + xName + ' sales ' + yName + ', ' + val + '.';
          }
      }
  },

  colorAxis: {
      min: 0,
      minColor: '#FFFFFF',
      maxColor: Highcharts.getOptions().colors[0]
  },

  tooltip: {
      formatter: function () {
          return '<b>' + getPointCategoryName(this.point, 'x') + '</b>  <br><b>' +
              getPointCategoryName(this.point, 'y') +':'+ this.point.value + '</b>';
      }
  },

  credits: {
    enabled: false,
  },
  legend:{
    enabled: false,
  },

  series: [{
      name: 'Reward',
      borderWidth: 1,
      data: [[0, 0, 0.12], [1, 0, 0.19], [2, 0, 0.08], [3, 0, -0.24], [4, 0, 0.067],[5, 0, 2.7], [6, 0, 1.7]],
  }],

exporting: {
  buttons: {
    contextButton: {
      menuItems: ["viewFullscreen", "printChart", "downloadCSV"],
    },
  },
},

}

  return (

    <div
      ref={heatMapWindow}
      className="highcharts-figure"
      >
      <HighchartsReact highcharts={Highcharts} options={options} />
    </div>
  );
};

export default ActionsHeatMap;

import React, { useState, useContext } from "react";
import classnames from "classnames";
import { Button, Offcanvas } from "react-bootstrap";
import Highcharts from "highcharts";
import HighchartsReact from "highcharts-react-official";
import ThemeContext from "../../context/ThemeContext";

const RewardHistory = () => {
  const [show, setShow] = useState(false);
  const { darkTheme } = useContext(ThemeContext);

  const handleClose = () => setShow(false);
  const handleShow = () => setShow(true);

  const options = {
    chart: {
      type: "area",
    },
    title: {
      text: "Reward History",
      style: {
        color: darkTheme && "white",
      },
    },
    xAxis: {
      categories: ["Step1", "Step2", "Step3", "Step4", "Step5"],
      labels: {
        style: {
          color: darkTheme && "white",
        },
      },
    },
    yAxis: {
      title: {
          text: 'Reward Value',
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
    credits: {
      enabled: false,
    },
    legend: {
      backgroundColor: 'white'
    },
    series: [
      {
        name: "Accumulated Reward",
        data: [5, 3, 4, 7, 2],
      },
      {
        name: "Reward",
        data: [2, -2, -3, 2, 1],
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
    <>
      <Button variant="primary" onClick={handleShow}>
        Reward History
      </Button>
      <Offcanvas show={show} onHide={handleClose} placement={"bottom"}>
        <Offcanvas.Header
          closeButton
          className={classnames(
            { "offcanvas-dark-mode": darkTheme },
            { "": darkTheme === false }
          )}
        >
          <Offcanvas.Title>Reward History</Offcanvas.Title>
        </Offcanvas.Header>
        <Offcanvas.Body className={classnames(
            { "offcanvas-dark-mode chart-dark-mode": darkTheme },
            { "": darkTheme === false }
          )
        }>
          <HighchartsReact highcharts={Highcharts} options={options} />
        </Offcanvas.Body>
      </Offcanvas>
    </>
  );
};

export default RewardHistory;

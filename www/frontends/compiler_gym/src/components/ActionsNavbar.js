import React, { useState, useEffect, useContext, forwardRef } from "react";
import {
  Form,
  FormControl,
  Col,
  InputGroup,
  Dropdown,
  Button,
} from "react-bootstrap";
import ApiContext from "../context/ApiContext";

const CustomMenu = forwardRef(
  ({ children, style, "aria-labelledby": labeledBy }, ref) => {
    const [value, setValue] = useState("");

    return (
      <div
        ref={ref}
        style={style}
        className="cg-dropdown"
        aria-labelledby={labeledBy}
      >
        <div className="search-bar">
          <FormControl
            autoFocus
            className="my-1"
            size="sm"
            placeholder="Search..."
            onChange={(e) => setValue(e.target.value)}
            value={value}
          />
        </div>
        <ul>
          {React.Children.toArray(children).filter(
            (child) =>
              !value || child.props.children.toLowerCase().startsWith(value)
          )}
        </ul>
      </div>
    );
  }
);

const ActionsNavbar = () => {
  const { session, compilerGym, api, setSession } = useContext(ApiContext);
  const [actionSpace, setActionSpace] = useState(15);
  const [actionsLine, setActionsLine] = useState("");
  const [benchmark, setBenchmark] = useState("benchmark://cbench-v1");
  const [uriOptions, setUriOptions] = useState([]);
  const [datasetUri, setDatasetUri] = useState("qsort");
  const [reward, setReward] = useState("IrInstructionCountOz");

  const benchmarkOptions =
    compilerGym.benchmarks &&
    Object.entries(compilerGym.benchmarks).map(([dataset, uri]) => ({
      dataset,
      uri,
    }));

  useEffect(() => {
    setActionsLine(session.state && session.state.commandline);
    return () => {};
  }, [session]);

  useEffect(() => {
    const selected =
      compilerGym.benchmarks &&
      Object.entries(compilerGym.benchmarks)
        .map(([dataset, uri]) => ({
          dataset,
          uri,
        }))
        .find((o) => o.dataset === benchmark);

    setUriOptions(selected && selected.uri);
    return () => {};
  }, [benchmark, compilerGym.benchmarks]);

  const handleBenchmarkSelect = (e) => {
    setBenchmark(e);
    setDatasetUri("");
  };

  const startNewSession = () => {
    let newBenchmark = `${benchmark}/${datasetUri}`;
    api.startSession(reward, newBenchmark).then(
      (result) => {
        setSession(result);
      },
      (error) => {
        console.log(error);
      }
    );
  };

  return (
    <div className="mx-2 action-navbar-wrapper">
      <Form>
        <Form.Row className="align-items-center">
          <Col sm={5} className="mt-1">
            <InputGroup className="mb-1">
              <Dropdown
                as={InputGroup.Prepend}
                onSelect={handleBenchmarkSelect}
              >
                <Dropdown.Toggle variant="dark" id="dropdown-benchmark">
                  Benchmark
                </Dropdown.Toggle>
                <Dropdown.Menu as={CustomMenu}>
                  {benchmarkOptions &&
                    benchmarkOptions.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i.dataset}
                        active={benchmark === i ? true : false}
                      >
                        {i.dataset}
                      </Dropdown.Item>
                    ))}
                </Dropdown.Menu>
              </Dropdown>
              <FormControl
                aria-describedby="basic-addon1"
                type="text"
                readOnly
                value={benchmark}
              />
            </InputGroup>
          </Col>
          <Col md={3} className="mt-1">
            <InputGroup className="mb-1">
              <Dropdown
                as={InputGroup.Prepend}
                onSelect={(e) => setDatasetUri(e)}
              >
                <Dropdown.Toggle variant="dark" id="dropdown-benchmark-uri">
                  Uri
                </Dropdown.Toggle>
                <Dropdown.Menu as={CustomMenu}>
                  {uriOptions &&
                    uriOptions.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i}
                        active={datasetUri === i ? true : false}
                      >
                        {i}
                      </Dropdown.Item>
                    ))}
                </Dropdown.Menu>
              </Dropdown>
              <FormControl
                aria-describedby="basic-addon1"
                type="text"
                readOnly
                value={datasetUri}
              />
            </InputGroup>
          </Col>
          <Col sm={4} className="mt-1">
            <InputGroup className="mb-1">
              <Dropdown as={InputGroup.Prepend} onSelect={(e) => setReward(e)}>
                <Dropdown.Toggle variant="dark" id="dropdown-reward">
                  Reward
                </Dropdown.Toggle>
                <Dropdown.Menu as={CustomMenu}>
                  {compilerGym.rewards &&
                    compilerGym.rewards.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i}
                        active={reward === i ? true : false}
                      >
                        {i}
                      </Dropdown.Item>
                    ))}
                </Dropdown.Menu>
              </Dropdown>
              <FormControl
                id="reward-input"
                aria-describedby="basic-addon3"
                type="text"
                readOnly
                value={reward}
              />
            </InputGroup>
          </Col>
        </Form.Row>
        <Form.Row className="align-items-center">
          <Col md={4} className="mt-1">
            <InputGroup className="mb-1">
              <InputGroup.Prepend>
                <InputGroup.Text
                  className="bg-dark"
                  id="inputGroup-sizing-sm"
                  style={{ color: "white" }}
                >
                  Action Space
                </InputGroup.Text>
              </InputGroup.Prepend>
              <FormControl
                aria-describedby="inputGroup-sizing-sm"
                type="number"
                placeholder="Select # action"
                min={1}
                name="actionSpace"
                value={actionSpace || 0}
                onChange={(e) => setActionSpace(e.target.value)}
              />
            </InputGroup>
          </Col>
          <Col md={1} className="mt-0 ml-auto">
            <Button size="sm" variant="success" onClick={startNewSession}>
              <i className="bi bi-play-fill"></i>
            </Button>
          </Col>
        </Form.Row>
        <Form.Row className="align-items-center">
          <Col sm={12} md={12} className="mt-1">
            <InputGroup className="mb-2">
              <InputGroup.Prepend>
                <InputGroup.Text
                  className="bg-dark"
                  id="inputGroup-sizing-sm"
                  style={{ color: "white" }}
                >
                  Actions
                </InputGroup.Text>
              </InputGroup.Prepend>
              <FormControl
                id="actions-input"
                type="text"
                aria-describedby="basic-addon1"
                readOnly
                defaultValue={actionsLine}
              />
            </InputGroup>
          </Col>
        </Form.Row>
      </Form>
    </div>
  );
};

export default ActionsNavbar;

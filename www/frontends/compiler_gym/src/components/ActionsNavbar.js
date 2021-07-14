import React, { useState, useEffect, useContext, forwardRef } from 'react';
import { Form, FormControl, Col, InputGroup, Dropdown } from 'react-bootstrap';
import ApiContext from '../context/ApiContext';

const benchmarkList = ["cbench-v1/qsort", "cenj-v2/sort34", "cbench-v1/crc32"];
const rewardList = [ "IrInstructionCountOz", "IrInstructionCountO3", "ObjectTextSizeO3", "ObjectTextSizeOz"]

const CustomMenu = forwardRef(
  ({ children, style, className, 'aria-labelledby': labeledBy }, ref) => {
    const [value, setValue] = useState('');

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
              !value || child.props.children.toLowerCase().startsWith(value),
          )}
        </ul>
      </div>
    );
  },
);


const ActionsNavbar = () => {
    const [actionSpace, setActionSpace] = useState(15);
    const [actionsLine, setActionsLine] = useState('')
    const { compilerGym } = useContext(ApiContext);

    useEffect(() => {
      setActionsLine(compilerGym.state && compilerGym.state.commandline)
      return () => {}
    }, [compilerGym])

    return (
      <div className="mx-2 action-navbar-wrapper">
        <Form>
          <Form.Row className="align-items-center">
            <Col sm={4} className="my-1">
              <InputGroup className="mb-1">
                <Dropdown as={InputGroup.Prepend}>
                  <Dropdown.Toggle variant="dark" id="dropdown-benchmark">
                    Benchmark
                  </Dropdown.Toggle>
                  <Dropdown.Menu as={CustomMenu}>
                    {benchmarkList && benchmarkList.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i}
                        //active={selected === i ? true : false}
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
                  defaultValue={"cbench-v1/qsort"}/>
              </InputGroup>
            </Col>
            <Col sm={4} className="my-1">
              <InputGroup className="mb-1">
                <InputGroup.Prepend>
                  <InputGroup.Text className="bg-dark" id="inputGroup-sizing-sm" style={{color:"white"}}>
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
                  onChange={e => setActionSpace(e.target.value)}
                />
              </InputGroup>
            </Col>
            <Col sm={4} className="my-1">
              <InputGroup className="mb-1">
                <Dropdown as={InputGroup.Prepend}>
                  <Dropdown.Toggle variant="dark" id="dropdown-reward">
                    Reward
                  </Dropdown.Toggle>
                  <Dropdown.Menu as={CustomMenu}>
                    {rewardList && rewardList.map((i, index) => (
                      <Dropdown.Item
                        key={index}
                        eventKey={i}
                        //active={selected === i ? true : false}
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
                  defaultValue={"IrInstructionCountOz"}/>
              </InputGroup>
            </Col>
          </Form.Row>
          <Form.Row className="align-items-center">
            <Col sm={12} md={12} className="my-0">
              <InputGroup className="mb-3">
                <InputGroup.Prepend>
                  <InputGroup.Text className="bg-dark" id="inputGroup-sizing-sm" style={{color:"white"}}>
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
    )
}

export default ActionsNavbar;

/**
 *
 * <InputGroup className="mb-3">
                <Dropdown as={InputGroup.Prepend}>
                  <Dropdown.Toggle variant="dark" id="input-group-dropdown-1">
                    Benchmark
                  </Dropdown.Toggle>
                  <Dropdown.Menu className="cg-dropdown">
                  {benchmarkList && benchmarkList.map((i, index) => (
                    <Dropdown.Item
                      key={index}
                      eventKey={i}
                      //active={selected === i ? true : false}
                    >
                      {i}
                    </Dropdown.Item>
                  ))}
                  </Dropdown.Menu>
                <FormControl aria-describedby="basic-addon1"/>
                </Dropdown>
              </InputGroup>
 */

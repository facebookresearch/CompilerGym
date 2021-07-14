import React, {useState, useContext} from 'react';
import { Row, Col, InputGroup, FormControl, Button } from 'react-bootstrap';
import ActionsNavbar from './ActionsNavbar';
import ApiContext from '../context/ApiContext';

const ControlsContainer = () => {

    const [stepID, setStepID] = useState('')
    const context = useContext(ApiContext);

    const handleSubmit = (e) => {
      e.preventDefault()
      context.submitStep(stepID)
    }

    return (
        <div>
          <ActionsNavbar/>
          <div className="search-tree-container">
              <p className="light">Hello from search tree</p>
              <br/>
              <Row>
                <Col md={3}>
                  <InputGroup className="mb-3">
                    <FormControl
                      placeholder="action_id"
                      aria-label="action_id"
                      aria-describedby="basic-addon2"
                      type="number"
                      value={stepID}
                      onChange={(e) => setStepID(e.target.value)}
                    />
                    <InputGroup.Append>
                      <Button
                        variant="secondary"
                        onClick={handleSubmit}
                        >
                          Submit
                      </Button>
                    </InputGroup.Append>
                  </InputGroup>
                </Col>
              </Row>
          </div>
        </div>
    )
}

export default ControlsContainer;

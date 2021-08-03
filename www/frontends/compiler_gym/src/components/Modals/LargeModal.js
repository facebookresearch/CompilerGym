/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useState, useRef } from "react";
import { Modal, Button, FormControl, Row, Col } from "react-bootstrap";

const LargeModal = ({ showModal, handleModal, title, content }) => {
  const [copySuccess, setCopySuccess] = useState("");
  const textAreaRef = useRef();

  const copyToClipboard = (e) => {
    textAreaRef.current.select();
    document.execCommand("copy");
    e.target.focus();
    setCopySuccess("Copied!");
  };

  return (
    <Modal size="lg" show={showModal} onHide={handleModal}>
      <Modal.Header closeButton>
        <Modal.Title>{title}</Modal.Title>
      </Modal.Header>
      <Modal.Body>
        <Row className="align-items-center">
          <Col lg={11} md={11} xs={10}>
            <FormControl
              aria-describedby="basic-addon1"
              ref={textAreaRef}
              type="text"
              readOnly
              value={content}
            />
          </Col>
          <Col lg={1} md={1} xs={1}>
            <i className="bi bi-clipboard cg-icon" onClick={copyToClipboard}></i>
          </Col>
        </Row>
        <Row className="justify-content-start">
          <Col lg={2} className="mt-1 h5">
            {copySuccess}
          </Col>
        </Row>
      </Modal.Body>
      <Modal.Footer>
        <Button variant="default" onClick={handleModal}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
};

export default LargeModal;

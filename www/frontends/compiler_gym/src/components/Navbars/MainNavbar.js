/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useContext } from "react";
import { Container, Navbar, Nav } from "react-bootstrap";
import ThemeContext from "../../context/ThemeContext";

const MainNavbar = () => {
  const { darkTheme, toggleTheme } = useContext(ThemeContext);

  return (
    <Navbar bg="dark" variant="dark" expand="lg">
      <Container fluid>
        <Navbar.Brand>CompilerGym Explorer</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link
              href="https://facebookresearch.github.io/CompilerGym/about.html"
              target="_blank"
            >
              About
            </Nav.Link>
            <Nav.Link
              href="https://github.com/facebookresearch/CompilerGym"
              target="_blank"
            >
              <i className="bi bi-github h2"></i>
            </Nav.Link>
            <Nav.Link onClick={toggleTheme}>
              {darkTheme ? (
                <i className="bi bi-brightness-high-fill h2" />
              ) : (
                <i className="bi bi-moon-fill h2" />
              )}
            </Nav.Link>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
  );
};

export default MainNavbar;

import React, { useContext} from 'react';
import { Container, Navbar, NavDropdown, Nav } from "react-bootstrap";
import ThemeContext from '../context/ThemeContext';

const MainNavbar = () => {

    const themeContext = useContext(ThemeContext)
    const { darkTheme, toggleTheme} = themeContext

    return (
			<Navbar bg="dark" variant="dark" expand="lg">
				<Container fluid>
					<Navbar.Brand href="#home">CompilerGym</Navbar.Brand>
					<Navbar.Toggle aria-controls="basic-navbar-nav" />
					<Navbar.Collapse id="basic-navbar-nav">
						<Nav className="mr-auto">
							<Nav.Link href="https://facebookresearch.github.io/CompilerGym/about.html" target="_blank">-</Nav.Link>
						</Nav>
						<Nav>
							<NavDropdown title="API" id="basic-nav-dropdown">
								<NavDropdown.Item href="https://facebookresearch.github.io/CompilerGym/" target="_blank">Docs</NavDropdown.Item>
								<NavDropdown.Divider />
								<NavDropdown.Item href="https://facebookresearch.github.io/CompilerGym/compiler_gym/compiler_gym.html" target="_blank">Python</NavDropdown.Item>
								<NavDropdown.Item href="https://facebookresearch.github.io/CompilerGym/cc/compiler_gym/envs/llvm/service.html" target="_blank">C++</NavDropdown.Item>
							</NavDropdown>
							<Nav.Link href="https://facebookresearch.github.io/CompilerGym/about.html" target="_blank">About</Nav.Link>
							<Nav.Link href="https://github.com/facebookresearch/CompilerGym" target="_blank">
								<i className="bi bi-github h2"></i>
							</Nav.Link>
							<Nav.Link onClick ={toggleTheme}>
                {darkTheme
                  ? <i className="bi bi-brightness-high-fill h2"/>
                  : <i className="bi bi-moon-fill h2"/> }
							</Nav.Link>
						</Nav>
					</Navbar.Collapse>
				</Container>
			</Navbar>

    )
}

export default MainNavbar

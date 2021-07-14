import React, { useEffect, useState } from "react";
import "./assets/scss/custom.scss";
import ApiService from "./api/ApiService";
import ApiContext from "./context/ApiContext";
import ThemeContext from "./context/ThemeContext";
import SplashPage from "./components/SplashPage";
import MainNavbar from "./components/MainNavbar";
import PanelsContainer from "./components/PanelsContainer";
import ControlsContainer from "./components/ControlsContainer";
import StateContainer from "./components/StateContainer";

const api = new ApiService("http://127.0.0.1:5000");

function App() {
  const [compilerGym, setCompilerGym] = useState({});
  const [darkTheme, setDarkTheme] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    setIsLoading(true);
    api.startSession().then(
      (result) => {
        console.log(result);
        setCompilerGym({ ...result });
        setIsLoading(false);
      },
      (error) => {
        console.log(error);
      }
    );
    return () => {};
  }, []);

  useEffect(() => {
    window.addEventListener("beforeunload", alertUser);
    window.addEventListener("unload", handleTabClosing);
    return () => {
      window.removeEventListener("beforeunload", alertUser);
      window.removeEventListener("unload", handleTabClosing);
    };
  });

  const handleTabClosing = () => {
    api.closeSession(compilerGym.session_id).then(
      (result) => {
        console.log(result);
      },
      (error) => {
        console.log(error);
      }
    );
  };

  const submitStep = (stepID) => {
    api.getStep(compilerGym.session_id, stepID).then(
      (result) => {
        console.log(result);
        setCompilerGym({ ...compilerGym, ...result });
      },
      (error) => {
        console.log(error);
      }
    );
  };

  const alertUser = (event) => {
    event.preventDefault();
    event.returnValue = "";
  };

  const toggleTheme = () => {
    setDarkTheme(!darkTheme);
  };

  if (isLoading) return <SplashPage />;

  return (
    <>
      <ApiContext.Provider
        value={{ compilerGym: compilerGym, submitStep: submitStep }}
      >
        <ThemeContext.Provider
          value={{ darkTheme: darkTheme, toggleTheme: toggleTheme }}
        >
          <div className="main-content">
            <MainNavbar />
            <PanelsContainer
              left={<ControlsContainer />}
              right={<StateContainer />}
            />
          </div>
        </ThemeContext.Provider>
      </ApiContext.Provider>
    </>
  );
}

export default App;

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React, { useEffect, useState } from "react";
import { BrowserRouter as Router, Switch, Route } from "react-router-dom";
import "./assets/scss/custom.scss";
import ApiService from "./api/ApiService";
import ApiContext from "./context/ApiContext";
import ThemeContext from "./context/ThemeContext";
import SplashPage from "./components/Pages/SplashPage";
import MainPage from "./components/Pages/MainPage";

const api = new ApiService();
const INITIAL_SETTINGS = {
  reward: "IrInstructionCountOz",
  dataset: "benchmark://cbench-v1",
  datasetUri: "adpcm",
};

function App() {
  const [compilerGym, setCompilerGym] = useState({});
  const [session, setSession] = useState({});
  const [darkTheme, setDarkTheme] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [params, setParams] = useState({
    reward: INITIAL_SETTINGS.reward,
    dataset: INITIAL_SETTINGS.dataset,
    datasetUri: INITIAL_SETTINGS.datasetUri,
  })

  /*
   * Start a new session when component mounts in the browser.
   * It collects CompilerGym variables
   */
  useEffect(() => {
    const fetchData = async () => {
      try {
        const options = await api.getEnvOptions();
        const initSession = await api.getActions(
          `${INITIAL_SETTINGS.dataset}/${INITIAL_SETTINGS.datasetUri}`,
          INITIAL_SETTINGS.reward,
          "",
        );
        setCompilerGym(options);
        setSession(initSession);
        setIsLoading(false);
      } catch (err) {
        console.log(err);
      }
    };

    setIsLoading(true);
    fetchData();
    return () => {};
  }, []);

  const toggleTheme = () => {
    setDarkTheme(!darkTheme);
  };

  if (isLoading) return <SplashPage />;

  return (
    <Router>
      <ApiContext.Provider
        value={{
          api: api,
          compilerGym: compilerGym,
          session: session,
          params: params,
          setParams,
          setSession,
        }}
      >
        <ThemeContext.Provider value={{ darkTheme: darkTheme, toggleTheme }}>
          <Switch>
            <Route path="/:session?">
              <MainPage />
            </Route>
          </Switch>
        </ThemeContext.Provider>
      </ApiContext.Provider>
    </Router>
  );
}

export default App;

/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

export default class ApiService {
  constructor(domain) {
    if (!process.env.NODE_ENV || process.env.NODE_ENV === "development") {
      this.domain = domain || "http://127.0.0.1:5000";
    } else {
      this.domain = domain || "http://production";
    }
    this.fetch = this.fetch.bind(this);
  }

  getEnvOptions() {
    return this.fetch(`${this.domain}/api/v4/describe`, {
      method: "GET",
    });
  }

  getActions(benchmark, reward, actionsIds, reward_history) {
    return this.fetch(
      `${this.domain}/api/v4/step?benchmark=${benchmark}&reward=${reward}&actions=${actionsIds}&reward_history=${reward_history}`,
      {
        method: "GET",
      }
    );
  }

  startSession(reward, actionsIds, benchmark) {
    return this.fetch(
      `${this.domain}/api/v3/start/${reward}/${actionsIds}/${benchmark}`,
      {
        method: "GET",
      }
    );
  }

  getSteps(sessionID, actionsIds) {
    return this.fetch(`${this.domain}/api/v3/step/${sessionID}/${actionsIds}`, {
      method: "GET",
    });
  }

  undoStep(sessionID, n) {
    return this.fetch(`${this.domain}/api/v3/undo/${sessionID}/${n}`, {
      method: "GET",
    });
  }

  closeSession(sessionID) {
    return this.fetch(`${this.domain}/api/v3/stop/${sessionID}`, {
      method: "GET",
    });
  }

  doFetch(url, options, includeCredentials = false) {
    const headers = {
      Accept: "application/json",
      "Content-Type": "application/json",
      Authorization: "None",
    };
    options = {
      headers,
      ...options,
    };
    if (includeCredentials) {
      options.credentials = "include";
    }
    return fetch(url).then(this.errorHandler);
  }

  fetch(url, options) {
    return this.doFetch(url, options, {}, true);
  }

  errorHandler(response) {
    try {
      if (response.status >= 200 && response.status < 300) {
        return Promise.resolve(response.json());
      } else {
        return Promise.resolve(response.json()).then((responseInJson) => {
          return Promise.reject(responseInJson);
        });
      }
    } catch (error) {
      console.log(error);
    }
  }
}

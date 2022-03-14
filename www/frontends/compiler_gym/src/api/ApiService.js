/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

export default class ApiService {
  constructor() {
    this.domain = process.env.REACT_APP_API_URL
    this.fetch = this.fetch.bind(this);
  }

  getEnvOptions() {
    return this.fetch(`${this.domain}/api/v4/describe`, {
      method: "GET",
    });
  }

  getActions(benchmark, reward, actionsIds, all_states) {
    return this.fetch(
      `${this.domain}/api/v4/step?benchmark=${benchmark}&reward=${reward}&actions=${actionsIds}&all_states=${all_states}`,
      {
        method: "GET",
      }
    );
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

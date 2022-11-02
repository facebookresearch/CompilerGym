# CompilerGym UI

This is the frontend to run compiler gym on the browser.

## Dependencies

In addition to the base requirements for [building from
source](/INSTALL.md#building-from-source), Install dependencies for the compiler gym explorer api with the `requirements.txt` from the current folder.

This also requires node. We recommend
[nvm](https://github.com/nvm-sh/nvm) to manage and install node versions.


## Usage

From the folder `frontends/compiler_gym`, Update the env files corresponding to the environment. `.env.development` & `.env.production` to configure API endpoints.

Build and launch the service on [http://localhost:5000/](http://localhost:5000/)
using:

```sh
make www
```

The port can be changed using the `$PORT` environment variable. For example, to
run the server on port 3000:

```sh
make www PORT=3000
```

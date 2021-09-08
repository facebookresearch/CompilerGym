# CompilerGym UI

This is the frontend to run compiler gym on the browser.

## Dependencies

In addition to the base requirements for [building from
source](/INSTALL.md#building-from-source), this requires node. We recommend
[nvm](https://github.com/nvm-sh/nvm) to manage and install node versions.


## Usage

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

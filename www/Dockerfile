# syntax=docker/dockerfile:1

# ####################
#  Web App - Build
# ####################
FROM node:16.8.0 as builder

ARG API_ENDPOINT

WORKDIR /usr/src/react_web_app
# copy npm package and lock file
COPY frontends/compiler_gym/package.json frontends/compiler_gym/package-lock.json ./
RUN --mount=type=cache,target=/usr/src/react_web_app/.npm --mount=type=cache,target=/usr/src/react_web_app/node_modules \
    npm install

# copy remaining UI code.
COPY frontends/compiler_gym/public public
COPY frontends/compiler_gym/src src

# configure api endpoint
RUN echo REACT_APP_API_URL=${API_ENDPOINT} > .env.production

# Compile frontend
RUN --mount=type=cache,target=/usr/src/react_web_app/.npm --mount=type=cache,target=/usr/src/react_web_app/node_modules \
    npm run build


FROM python:3.8-slim-buster

WORKDIR /app

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libtinfo5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create an unversioned library for libtinfo5 so that -ltinfo works.
RUN ln -s /lib/x86_64-linux-gnu/libtinfo.so.5 /lib/x86_64-linux-gnu/libtinfo.so

# Install www dependencies.
COPY requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Run the LLVM environment now to warm the CompilerGym caches.
RUN python3 -m compiler_gym.bin.service --env=llvm-v0

COPY www.py .
COPY --from=builder /usr/src/react_web_app/build frontends/compiler_gym/build

CMD [ "python3", "www.py"]

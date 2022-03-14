# syntax=docker/dockerfile:1

# ####################
#  React Web App - Build
# ####################
FROM node:16.8.0 as builder

WORKDIR /usr/src/app

# Copy npm package info
COPY frontends/compiler_gym/package.json frontends/compiler_gym/package-lock.json ./

# Instal npm packages
RUN --mount=type=cache,target=/usr/src/app/.npm --mount=type=cache,target=/usr/src/app/node_modules \
    npm install

# Copy source
COPY frontends/compiler_gym/public public
COPY frontends/compiler_gym/src src
COPY frontends/compiler_gym/.env.production .

# Build Ract app
RUN --mount=type=cache,target=/usr/src/app/.npm --mount=type=cache,target=/usr/src/app/node_modules \
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
COPY --from=builder /usr/src/app/build frontends/compiler_gym/build

CMD [ "python3", "www.py"]

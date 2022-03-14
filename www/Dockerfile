# syntax=docker/dockerfile:1

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
COPY frontends/compiler_gym/build frontends/compiler_gym/build

CMD [ "python3", "www.py"]

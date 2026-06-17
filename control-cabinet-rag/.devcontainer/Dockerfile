FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

WORKDIR /workspaces/control-cabinet-rag

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

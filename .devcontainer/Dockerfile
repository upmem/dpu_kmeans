# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.194.3/containers/ubuntu/.devcontainer/base.Dockerfile

# [Choice] Ubuntu version: hirsute, bionic, focal
ARG VARIANT="focal"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

RUN wget http://sdk-releases.upmem.com/2021.3.0/ubuntu_20.04/upmem_2021.3.0_amd64.deb \
    && sudo apt update \
    && sudo apt install -y ./upmem_2021.3.0_amd64.deb \
    && rm upmem_2021.3.0_amd64.deb
RUN sudo apt update \
    && sudo apt install -y build-essential cmake ninja-build python-is-python3 \
    && sudo apt full-upgrade -y

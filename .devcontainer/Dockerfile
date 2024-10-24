# See here for image contents: https://github.com/devcontainers/images/tree/main/src/base-ubuntu

ARG VARIANT="jammy"
FROM mcr.microsoft.com/vscode/devcontainers/base:0-${VARIANT}

# [Optional] Uncomment this section to install additional OS packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install build-essential ninja-build cppcheck nox pre-commit\
    && apt-get dist-upgrade -y
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && wget http://sdk-releases.upmem.com/2024.2.0/ubuntu_22.04/upmem_2024.2.0_amd64.deb \
    && apt-get install -y ./upmem_2024.2.0_amd64.deb \
    && rm upmem_2024.2.0_amd64.deb

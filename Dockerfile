FROM python:3.9-slim AS python_base
LABEL maintainer="Daniel Lehmberg <d.lehmberg@tum.de>"

# Set environment variable for datafold's Makefile to *not* create a second layer of
# virtualization with Python/virtualenv
ENV DOCKER_ENVIRONMENT=true

ARG DATAFOLD_ENV_PATH="/home/datafold_env_files"
ARG DATAFOLD_MOUNT_DIR="/home/datafold/"

RUN mkdir "$DATAFOLD_ENV_PATH"
RUN mkdir "${DATAFOLD_MOUNT_DIR}"
WORKDIR "${DATAFOLD_ENV_PATH}"

# install apt dependencies to build the docs first
ADD Makefile ./
RUN apt-get update \
    && apt-get -y --no-install-recommends install git make dialog apt-utils \
    && make install_docdeps \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    # command "latex" is used to build math equations in the documentation (see configuration in "conf.py").
    # Point "latex" to execute "pdftex":
    && ln -sf /usr/bin/pdftex /usr/bin/latex


# install datafold dependencies
ADD requirements.txt "${DATAFOLD_ENV_PATH}"

RUN make install_deps \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*;


# install datafold-dev dependencies
ADD requirements-dev.txt "${DATAFOLD_ENV_PATH}"

RUN make install_devdeps \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*;


WORKDIR "${DATAFOLD_MOUNT_DIR}"

ENTRYPOINT [ "/bin/bash" ]

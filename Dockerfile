FROM python:3.7-bullseye
MAINTAINER Daniel Lehmberg (daniel.lehmberg@hm.edu)

ARG gl_username=datafold-dev
ARG datafold_cont="/home/datafold-container"
ARG branch=master

ENV DOCKER_ENVIRONMENT=true

RUN mkdir "$datafold_cont"
WORKDIR "$datafold_cont"

# clone (forked) gitlab repository
RUN git clone "https://gitlab.com/$gl_username/datafold.git" "$datafold_cont" \
    && cd "$datafold_cont" \
	&& git config pull.rebase false \
	# set original datafold repository as "upstream" (besides "origin")
    && git remote add upstream "https://gitlab.com/datafold-dev/datafold.git" \
    && git pull upstream master \
    && git checkout "$branch"

ADD Makefile "$datafold_cont"

RUN apt-get update \
    && apt-get -y install dialog apt-utils

RUN cd "$datafold_cont" \
    && make install_docdeps \
    # cleanup
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*;

# command "latex" is used to build math equations in the documentation (see configuration in "conf.py").
# Point "latex" to execute "pdftex":
RUN ln -sf /usr/bin/pdftex /usr/bin/latex

# Install development dependencies according to make
RUN cd "$datafold_cont" \
    && make install_devdeps;

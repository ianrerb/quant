FROM python:3.6-stretch

# https://github.com/phusion/baseimage-docker/issues/58
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
  locales \
  && rm -rf /var/lib/apt/lists/* \
  && locale-gen "en_US.UTF-8" \
  && localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

ENV LC_ALL="en_US.UTF-8" LANG="en_US.utf8"

RUN apt-get update && apt-get install -y --no-install-recommends \
  apt-transport-https \
  apt-utils \
  bash-completion \
  ca-certificates \
  curl \
  dnsutils \
  freetype* \
  g++ \
  gfortran \
  git-core \
  gnupg \
  gpg-agent \
  graphviz \
  graphviz-dev \
  less \
  libblas-dev \
  libffi-dev \
  libgeos-dev \
  libgraphviz-dev \
  liblapack-dev \
  libproj-dev \
  libssl-dev \
  libyaml-dev \
  netcat \
  pkg-config \
  proj-bin \
  unixodbc \
  unixodbc-dev \
  unzip \
  vim \
  wget \
  && rm -rf /var/lib/apt/lists/*

# docker (simple version), from https://docs.docker.com/engine/installation/linux/ubuntu
RUN curl -fsSL get.docker.com | sh

# standard requirements
COPY requirements.txt /
RUN pip install pip --upgrade \
  # https://github.com/un33k/python-slugify/issues/52
  && pip install --no-cache -r requirements.txt \
  # TODO: always return true until conflicts are resolved
  && pip check || true
# https://github.com/antocuni/pdb/issues/58
# remove at Py3
RUN pip uninstall -y pyrepl

# Jupyter & kernel gateway
ENV JUPYTER_USE_HTTPS='1' JUPYTER_NOTEBOOK_DIR='/notebooks'
# can't use env vars in `COPY`, so use two steps
COPY jupyter_notebook_config.py .jupyter/
RUN mv .jupyter $HOME/.jupyter
RUN jupyter notebook list

# Mosek
ENV MOSEKLM_LICENSE_FILE /mosek.lic
RUN pip install -f https://download.mosek.com/stable/wheel/index.html "Mosek<9.0.0" \
  && python -c '__import__("mosek").Env()'
COPY mosek.lic /mosek.lic

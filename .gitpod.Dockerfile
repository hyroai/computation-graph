FROM python/3.9-bullseye

USER gitpod

RUN sudo apt-get install -yq graphviz graphviz-dev python3-dev
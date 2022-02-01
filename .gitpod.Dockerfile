FROM python:3.9-bullseye
USER root
RUN apt install -yq graphviz graphviz-dev python3-dev
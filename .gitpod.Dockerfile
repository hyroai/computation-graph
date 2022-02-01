FROM gitpod/workspace-full

USER gitpod

RUN sudo apt-get install -yq graphviz graphviz-dev python3-dev && pyenv install 3.9.10  && pyenv global 3.9.10
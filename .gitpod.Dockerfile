FROM  gitpod/workspace-full

USER gitpod

RUN sudo apt-get install -yq python3-dev graphviz graphviz-dev && \
    pyenv update  && \
    pyenv install 3.9.10  && \
    pyenv global 3.9.10  && \
    python -m pip install --no-cache-dir --upgrade pip && \
    echo "alias pip='python -m pip'" >> ~/.bash_aliases

FROM debian:11.5
WORKDIR /home/llmvm/llmvm

# we grab the keys from the terminal environment
# use the --build-arg flag to pass in the keys
ARG OPENAI_API_KEY
ARG ANTHROPIC_API_KEY
ARG SEC_API_KEY
ARG SERPAPI_API_KEY

ENV container docker
ENV PATH "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

RUN useradd -m -d /home/llmvm -s /bin/bash -G sudo llmvm
RUN mkdir -p /var/run/sshd
RUN mkdir -p /run/sshd
RUN mkdir -p /tmp

# some of these aren't acually required.
RUN apt-get update
RUN apt-get install -y dialog apt-utils
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
RUN apt-get install -y git
RUN apt-get install -y wget
RUN apt-get install -y vim
RUN apt-get install -y dpkg
RUN apt-get install -y build-essential

RUN apt-get install -y linux-headers-generic
RUN apt-get install -y lzip curl
RUN apt-get install -y locales-all
RUN apt-get install -y redis-server
RUN apt-get install -y openssh-server
RUN apt-get install -y sudo
RUN apt-get install -y unzip
RUN apt-get install -y tmux
RUN apt-get install -y expect
RUN apt-get install -y iproute2
RUN apt-get install -y net-tools
RUN apt-get install -y rsync
RUN apt-get install -y iputils-ping
RUN apt-get install -y lnav
RUN apt-get install -y poppler-utils
RUN apt-get install -y gosu

# required for pyenv to build 3.10.11 properly
RUN apt-get install -y libbz2-dev
RUN apt-get install -y libsqlite3-dev
RUN apt-get install -y libreadline-dev

# python version upgrade
RUN apt-get install -y python3-venv
RUN apt-get install -y libedit-dev
RUN apt-get install -y libncurses5-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y libffi-dev
RUN apt-get install -y liblzma-dev
RUN apt-get install -y firefox-esr

RUN echo 'llmvm:llmvm' | chpasswd
RUN service ssh start

# ssh
EXPOSE 22
# server.py
EXPOSE 8000

# copy over the source and data
COPY ./ /home/llmvm/llmvm/

RUN apt-get update -y

RUN mkdir /home/llmvm/.config
RUN mkdir /home/llmvm/.config/llmvm
RUN mkdir /home/llmvm/.tmp
RUN mkdir /home/llmvm/.cache
RUN mkdir /home/llmvm/.local
RUN mkdir /home/llmvm/.local/share
RUN mkdir /home/llmvm/.local/share/llmvm
RUN mkdir /home/llmvm/.local/share/llmvm/cache
RUN mkdir /home/llmvm/.local/share/llmvm/download
RUN mkdir /home/llmvm/.local/share/llmvm/cdn
RUN mkdir /home/llmvm/.local/share/llmvm/logs
RUN mkdir /home/llmvm/.local/share/llmvm/faiss
RUN mkdir /home/llmvm/.ssh

RUN chown -R llmvm:llmvm /home/llmvm
RUN chsh -s /bin/bash llmvm

# install all the python 3.11.7 runtimes and packages
USER llmvm

ENV HOME /home/llmvm
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$HOME/.local/bin:$PATH
ENV TMPDIR $HOME/.tmp

# setup defaults for bash
RUN echo 'if [ -f ~/.bashrc ]; then\n   source ~/.bashrc\nfi' >> /home/llmvm/.bash_profile

# Set default PATH in .bashrc
RUN echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> /home/llmvm/.bashrc

# Initialize pyenv in .bashrc
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> /home/llmvm/.bashrc && \
    echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> /home/llmvm/.bashrc && \
    echo 'eval "$(pyenv init --path)"' >> /home/llmvm/.bashrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /home/llmvm/.bashrc

# Ensure pyenv runs on directory change in .bashrc
RUN echo 'if command -v pyenv 1>/dev/null 2>&1; then' >> /home/llmvm/.bashrc && \
    echo '  eval "$(pyenv init --path)"' >> /home/llmvm/.bashrc && \
    echo 'fi' >> /home/llmvm/.bashrc

RUN echo 'cd llmvm' >> /home/llmvm/.bashrc

RUN curl https://pyenv.run | bash

WORKDIR /home/llmvm/llmvm

RUN pyenv install 3.11.7
RUN pyenv virtualenv 3.11.7 llmvm
ENV PYENV_VERSION llmvm
WORKDIR /home/llmvm/llmvm
RUN pyenv local llmvm

RUN python3 --version
RUN pip install -r requirements.txt

COPY ./llmvm/config.yaml /home/llmvm/.config/llmvm/config.yaml

# change this to GPU if you have a GPU
RUN pip install faiss-cpu

# ARG OPENAI_API_KEY
# ARG ANTHROPIC_API_KEY
# ARG MISTRAL_API_KEY
# ARG GOOGLE_API_KEY
# ARG SEC_API_KEY
# ARG SERPAPI_API_KEY

# ENV OPENAI_API_KEY=$OPENAI_API_KEY
# ENV ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
# ENV MISTRAL_API_KEY=$MISTRAL_API_KEY
# ENV GOOGLE_API_KEY=$GOOGLE_API_KEY
# ENV SEC_API_KEY=$SEC_API_KEY
# ENV SERPAPI_API_KEY=$SERPAPI_API_KEY

RUN echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "SEC_API_KEY=$SEC_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "SERPAPI_API_KEY=$SERPAPI_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "MISTRAL_API_KEY=$MISTRAL_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "GOOGLE_API_KEY=$GOOGLE_API_KEY" >> /home/llmvm/.ssh/environment

RUN playwright install firefox

# spin back to root, to start sshd
USER root

RUN echo 'PermitUserEnvironment yes' >> /etc/ssh/sshd_config

WORKDIR /home/llmvm/llmvm
ENTRYPOINT service ssh restart && sudo -Eu llmvm /usr/bin/python -m llmvm.server.server && tail -f /dev/null

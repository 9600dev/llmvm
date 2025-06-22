FROM debian:11.5
WORKDIR /home/llmvm/llmvm

# we grab the keys from the terminal environment
# use the --build-arg flag to pass in the keys
ARG OPENAI_API_KEY
ARG ANTHROPIC_API_KEY
ARG GEMINI_API_KEY
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
RUN apt-get install -y openssh-server
RUN apt-get install -y sudo
RUN apt-get install -y unzip
RUN apt-get install -y expect
RUN apt-get install -y iproute2
RUN apt-get install -y net-tools
RUN apt-get install -y rsync
RUN apt-get install -y iputils-ping
RUN apt-get install -y lnav
RUN apt-get install -y poppler-utils
RUN apt-get install -y gosu

# required for building Python and some packages
RUN apt-get install -y libbz2-dev
RUN apt-get install -y libsqlite3-dev
RUN apt-get install -y libreadline-dev
RUN apt-get install -y libedit-dev
RUN apt-get install -y libncurses5-dev
RUN apt-get install -y libssl-dev
RUN apt-get install -y libffi-dev
RUN apt-get install -y liblzma-dev

# Install Node.js and nginx for the web application
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
RUN apt-get install -y nodejs
RUN apt-get install -y nginx

RUN echo 'llmvm:llmvm' | chpasswd
RUN service ssh start

# ssh
EXPOSE 2222
# llmvm.server
EXPOSE 8011
# website
EXPOSE 8080

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
RUN mkdir /home/llmvm/.local/share/llmvm/logs
RUN mkdir /home/llmvm/.local/share/llmvm/memory
RUN mkdir /home/llmvm/.ssh

RUN chown -R llmvm:llmvm /home/llmvm
RUN chsh -s /bin/bash llmvm

# Create a separate SSH config for standard shell access
RUN mkdir -p /etc/ssh/sshd_config.d
RUN cp /etc/ssh/sshd_config /etc/ssh/sshd_config_standard
RUN sed -i 's/^#Port 22/Port 2222/' /etc/ssh/sshd_config_standard
RUN sed -i '/Match User llmvm/,/ForceCommand/d' /etc/ssh/sshd_config_standard
RUN echo 'PidFile /var/run/sshd_standard.pid' >> /etc/ssh/sshd_config_standard

# install miniconda and Python 3.13.2
USER llmvm

ENV HOME /home/llmvm
ENV TMPDIR $HOME/.tmp

# Detect architecture and download appropriate Miniconda installer
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget -q "$MINICONDA_URL" -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p $HOME/miniconda && \
    rm ~/miniconda.sh

# Add conda to PATH
ENV PATH=$HOME/miniconda/bin:$PATH

# Initialize conda for bash
RUN conda init bash

# setup defaults for bash
RUN echo 'if [ -f ~/.bashrc ]; then\n   source ~/.bashrc\nfi' >> /home/llmvm/.bash_profile

# Add conda initialization to .bashrc
RUN echo 'export PATH="$HOME/miniconda/bin:$PATH"' >> /home/llmvm/.bashrc
RUN echo 'cd llmvm' >> /home/llmvm/.bashrc

WORKDIR /home/llmvm/llmvm

# Create conda environment with Python 3.13.2
RUN conda create -n llmvm python=3.13.2 -y

# Activate the environment
SHELL ["conda", "run", "-n", "llmvm", "/bin/bash", "-c"]

# Install requirements
RUN conda run -n llmvm pip install -r requirements.txt
RUN conda run -n llmvm pip install playwright

COPY ./llmvm/config.yaml /home/llmvm/.config/llmvm/config.yaml

RUN echo "OPENAI_API_KEY=$OPENAI_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "GEMINI_API_KEY=$GEMINI_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "SEC_API_KEY=$SEC_API_KEY" >> /home/llmvm/.ssh/environment
RUN echo "SERPAPI_API_KEY=$SERPAPI_API_KEY" >> /home/llmvm/.ssh/environment

RUN conda run -n llmvm bash -c "playwright install"

# Build the SDK first
WORKDIR /home/llmvm/llmvm/web/js-llmvm-sdk
RUN npm install
RUN npm run build

# Build the website
WORKDIR /home/llmvm/llmvm/web/llmvm-chat-studio
# Add the SDK as a local dependency and install
RUN npm install ../js-llmvm-sdk
RUN npm install
RUN npm run build

# Copy the wrapper script from the scripts directory and make it executable
WORKDIR /home/llmvm/llmvm
COPY --chmod=755 ./scripts/llmvm-client-wrapper.sh /home/llmvm/llmvm-client-wrapper.sh

# spin back to root, to start sshd
USER root

# Configure nginx
COPY ./docker/nginx.conf /etc/nginx/sites-available/llmvm-web
RUN ln -s /etc/nginx/sites-available/llmvm-web /etc/nginx/sites-enabled/
RUN rm -f /etc/nginx/sites-enabled/default

RUN sed -i 's/^#Port 22/Port 2222/' /etc/ssh/sshd_config
RUN echo 'PermitUserEnvironment yes' >> /etc/ssh/sshd_config
RUN echo 'PermitUserEnvironment yes' >> /etc/ssh/sshd_config_standard

# Configure SSH to use the wrapper script as the shell for the llmvm user
RUN echo 'Match User llmvm' >> /etc/ssh/sshd_config_standard && \
    echo '    ForceCommand /home/llmvm/llmvm-client-wrapper.sh' >> /etc/ssh/sshd_config_standard

WORKDIR /home/llmvm/llmvm

ENTRYPOINT service ssh restart; \
    service nginx start; \
    /usr/sbin/sshd -f /etc/ssh/sshd_config; \
    /usr/sbin/sshd -f /etc/ssh/sshd_config_standard; \
    sudo -E -u llmvm bash -c 'source ~/.bashrc; cd /home/llmvm/llmvm; conda activate llmvm; LLMVM_FULL_PROCESSING="true" LLMVM_EXECUTOR_TRACE="~/.local/share/llmvm/executor.trace" LLMVM_PROFILING="true" python -m llmvm.server.server'

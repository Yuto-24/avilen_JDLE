FROM nvidia/cuda:12.2.0-devel-ubuntu20.04

ENV http_proxy=http://proxy.otsuka-shokai.co.jp:8080
ENV https_proxy=http://proxy.otsuka-shokai.co.jp:8080

## apt install
RUN apt update && \
    apt install -y \
    git \
    curl \
    wget \
    iputils-ping \
    net-tools \
    uvicorn

# Set environment to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install Python
RUN apt update -q && apt install -yq software-properties-common && \
    yes | add-apt-repository ppa:deadsnakes/ppa && \
    apt remove -yq python3.8 && \
    apt install -yq python3.10 python3-pip && \
    ln -fs /usr/bin/python3.10 /usr/bin/python3


RUN apt --fix-missing update && \
    apt install -y --no-install-recommends build-essential && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    pip3 install -U pip && \
    pip3 install -U setuptools && \
    apt install python3-setuptools

# Install Package
WORKDIR /root
COPY requirements.txt /root

RUN pip3 install -r requirements.txt


# エイリアスの追加
RUN echo "alias ll='ls -al'" >> ~/.bashrc && \
    echo "alias python='python3'" >> ~/.bashrc && \
    echo "alias pip='pip3'" >> ~/.bashrc && \
    echo "source ~/.bashrc" > ~/.bash_profile

# コンテナログイン時のディレクトリ指定
WORKDIR /projects/

ADD ./run.sh /run.sh
RUN chmod +x /run.sh \
    && export LC_ALL=C.UTF-8

ADD ./run.sh /root/run.sh
RUN chmod +x /root/run.sh
CMD [ "sh", "-c", "/root/run.sh" ]

# FROM nvidia/cuda:11.5.1-devel-ubuntu20.04
FROM nvidia/cuda:11.5.0-cudnn8-devel-ubuntu20.04

USER root
ADD . /ins_env
WORKDIR /ins_env

ENV http_proxy=http://proxy.otsuka-shokai.co.jp:8080
ENV https_proxy=http://proxy.otsuka-shokai.co.jp:8080

# nvidia-cuda-toolkitのパスを通す.
ENV PATH="/usr/local/cuda-11.5/bin$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.5/lib64:$LD_LIBRARY_PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.5/targets:$LD_LIBRARY_PATH"
ENV DEBIAN_FRONTEND=noninteractive

RUN export http_proxy=http://proxy.otsuka-shokai.co.jp:8080 \
    && export https_proxy=http://proxy.otsuka-shokai.co.jp:8080 \
    && apt-get -y update \
    && apt-get -y upgrade \
    && apt-get install -y git \
    && apt-get install -y --fix-missing make \
    && apt-get install -y --fix-missing wget \
    && apt-get install -y --fix-missing curl \
    && apt-get install -y --fix-missing libsqlite3-dev libreadline6-dev libbz2-dev libssl-dev libsqlite3-dev libncursesw5-dev libffi-dev libdb-dev libexpat1-dev zlib1g-dev liblzma-dev libgdbm-dev libmpdec-dev\
    && apt-get install -y --fix-missing mecab libmecab-dev mecab-utils mecab-jumandic-utf8 mecab-naist-jdic python3-mecab


RUN apt-get install -y git


#python 3.10.13のダウンロード
WORKDIR /root/
RUN wget https://www.python.org/ftp/python/3.10.13/Python-3.10.13.tar.xz \
    && tar xvf Python-3.10.13.tar.xz \
    && cd Python-3.10.13 \
    && ls -al | sed 10q \
    && ./configure --enable-optimizations --with-lto \
    && make \
    && make install
RUN rm Python-3.10.13.tar.xz

# pipのダウンロード
WORKDIR /root/Python-3.10.13
RUN ln -fs /root/Python-3.10.13/python /usr/bin/python \
    && curl -kL https://bootstrap.pypa.io/get-pip.py | python \
    && rm -rf /var/lib/apt/lists/*

RUN alias python="python3" \
    && alias pip="pip3"

# python 3.10.13インストール後にライブラリをインストール.
WORKDIR /ins_env
RUN pip3 install --proxy="http://proxy.otsuka-shokai.co.jp:8080/" --upgrade pip \
    && pip3 install --proxy="http://proxy.otsuka-shokai.co.jp:8080/" -r ./requirements.txt \
    # mecab用の辞書のダウンロード
    && pip install unidic

RUN export http_proxy=http://proxy.otsuka-shokai.co.jp:8080 \
    && export https_proxy=http://proxy.otsuka-shokai.co.jp:8080 \
    && jupyter notebook --generate-config

# コンテナログイン時のディレクトリ指定
WORKDIR /projects/

ADD ./run.sh /run.sh
RUN chmod +x /run.sh \
    && export LC_ALL=C.UTF-8

CMD ["/run.sh"]

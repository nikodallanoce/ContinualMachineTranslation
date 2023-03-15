FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
  && apt-get install -y build-essential \
      gcc \
      g++ \
      gdb \
      clang \
      make \
      ninja-build \
      cmake \
      autoconf \
      automake \
      locales-all \
      dos2unix \
      rsync \
      tar \
      python \
      python-dev \
      cmake\
      libopenblas-dev\
      liblapack-dev\
      libarpack2-dev\
      libsuperlu-dev\
      wget\
  && apt-get clean

RUN apt-get update \
  && apt-get install -y libboost-all-dev\
    libeigen3-dev\
    bzip2\
    liblzma-dev\
    zip\
    python3-pip\
    git\
    python-is-python3\
    && apt-get clean

RUN git clone https://github.com/facebookresearch/cc_net.git
WORKDIR /cc_net
RUN make install
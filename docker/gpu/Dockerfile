FROM tensorflow/tensorflow:1.14.0-gpu

RUN \
  apt-get update && \
  apt-get install -y \
    # stack needs git
    git \
    # Required by snappy-frames dependency.
    libsnappy-dev \
    # Avoids /usr/bin/ld: cannot find -ltinfo
    libncurses5-dev \
    # Makes stack viable in the container
    libgmp-dev \
    # Required for locales configuration.
    locales

# Support output of Unicode characters.
RUN \
  dpkg-reconfigure locales && \
  locale-gen en_US.UTF-8 && \
  update-locale LANG=en_US.UTF-8

RUN \
  # protoc
  curl -O -L https://github.com/google/protobuf/releases/download/v3.8.0/protoc-3.8.0-linux-x86_64.zip && \
  unzip -d /usr/local protoc-3.8.0-linux-x86_64.zip bin/protoc && \
  chmod 755 /usr/local/bin/protoc && \
  rm protoc-3.8.0-linux-x86_64.zip && \
  # libtensorflow
  curl -O https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz && \
  tar zxf libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz -C /usr/local && \
  ldconfig && \
  rm libtensorflow-gpu-linux-x86_64-1.14.0.tar.gz

ENV LANG en_US.UTF-8

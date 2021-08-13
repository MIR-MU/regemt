FROM nvidia/cuda:11.0-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    TERM=xterm

COPY . /app/mt-eval/

RUN rm -rf /app/mt-eval/submit_dir \
 && ln -s /submit_dir /app/mt-eval/submit_dir \
 && apt-get -qy update \
 && apt-get -qy install --no-install-recommends \
    build-essential \
    curl \
    git \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
 && curl https://bootstrap.pypa.io/get-pip.py | python3.8 \
 && python3 -m pip install -r /app/mt-eval/requirements.txt \
 && python3 -m pip install -U numpy \
 && apt-get -qy remove --purge \
    build-essential \
    curl \
    git \
    python3.8-dev \
    python3.8-distutils \
 && apt-get -qy autoclean \
 && apt-get -qy clean \
 && apt-get -qy autoremove --purge \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /root/.cache/pip

WORKDIR /app/mt-eval
ENTRYPOINT ["python3", "-m", "main"]

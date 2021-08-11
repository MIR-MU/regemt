FROM nvidia/cuda:11.0

# must-have tools in each docker image in production \
RUN apt-get update && apt-get install -y python3.8 python3.8-distutils python3.8-dev \
    bash less netcat vim curl wget nmap traceroute net-tools iputils-ping openssh-client \
    telnet strace mc dnsutils procps htop build-essential git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN curl https://bootstrap.pypa.io/get-pip.py | python3.8

COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=secret,id=netrc,dst=/root/.netrc \
    python3.8 -m pip install -r /tmp/requirements.txt

COPY ../mt-eval/ /app/mt-eval/

WORKDIR /app/mt-eval
CMD ["python3.8", "-m", "main"]

FROM --platform=linux/amd64 python:3.6.10-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /work

RUN pip install --no-cache-dir \
    numpy==1.16.4 \
    scipy==1.2.3 \
    pillow==8.4.0 \
    pyyaml==5.4.1 \
    networkx==1.11 \
    protobuf==3.19.6 \
    tensorflow==1.13.1

ENV PYTHONPATH=/work/third_party/convrnns:/work/python_scripts/src

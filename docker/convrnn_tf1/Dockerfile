FROM --platform=linux/amd64 python:3.7-slim-buster

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /work

RUN pip install --no-cache-dir \
    numpy==1.18.5 \
    scipy==1.4.1 \
    pillow==9.5.0 \
    pyyaml==6.0.1 \
    networkx==1.11 \
    protobuf==3.20.3 \
    tensorflow==1.15.5

ENV PYTHONPATH=/work/third_party/convrnns:/work/python_scripts/src

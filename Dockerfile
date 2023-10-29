# syntax=docker/dockerfile:1
FROM pytorch/pytorch@sha256:cf9197f9321ac3f49276633b4e78c79aa55f22578de3b650b3158ce6e3481f61

ADD /requirements.txt /docker_data/requirements.txt
RUN pip3 install -r /docker_data/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD /src /docker_data/src/
ADD /runs /docker_data/runs/

WORKDIR /docker_data/
ENTRYPOINT ["python3", "-u", "/docker_data/src/main.py"]
# syntax=docker/dockerfile:1
FROM pytorch/pytorch@sha256:e4aaefef0c96318759160ff971b527ae61ee306a1204c5f6e907c4b45f05b8a3

ADD /requirements.txt /docker_data/requirements.txt
RUN pip3 install -r /docker_data/requirements.txt

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD /src /docker_data/src/
ADD /runs /docker_data/runs/

WORKDIR /docker_data/
ENTRYPOINT ["python3", "-u", "/docker_data/src/main.py"]
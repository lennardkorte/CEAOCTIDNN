# syntax=docker/dockerfile:1
FROM pytorch/pytorch@sha256:b32443a58a60c4ca4d10651e3c5b2aa7211ffcfbdc433593d71b53c2f2e94703

ADD /requirements.txt /docker_data/requirements.txt
RUN pip3 install -r /docker_data/requirements.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y libglib2.0-0

# Original bigger packages:
# RUN apt-get install ffmpeg libsm6 libxext6  -y

ADD /src /docker_data/src/
ADD /runs /docker_data/runs/

WORKDIR /docker_data/
ENTRYPOINT ["python3", "-u", "/docker_data/src/main.py"]
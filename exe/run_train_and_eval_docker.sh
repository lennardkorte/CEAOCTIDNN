#! /bin/bash

green=`tput setaf 2`
reset=`tput sgr0`

date=$(date '+%Yy-%mm-%dd_%Hh-%Mm-%Ss')
name_image="image-${date}"
name_container="container-${date}"
filename="train_and_test.py"
data_path="/docker_data/data"

echo -e "${green}\n\nBuilding docker-image...${reset}"
docker build -t $name_image .

echo -e "${green}\n\nRemoving additional <none> images...${reset}"
docker rm $(docker ps -a -q) > /dev/null 2>&1
docker image prune -f

echo -e "${green}\n\nShow all images:${reset}"
docker image ls

echo -e "${green}\n\nRun docker-image:${reset}"
args="$@"
s_path="${PWD}/data"
docker run \
-it --rm \
--gpus all \
--shm-size 8G \
--name $name_container \
--mount type=bind,source=$s_path,target=$data_path \
-i $name_image "src/${filename} ${args}"


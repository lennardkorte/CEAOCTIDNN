#! /bin/bash

green=`tput setaf 2`
reset=`tput sgr0`

name_image="image-iddatdloct"
name_container="container-iddatdloct"
filename="train_and_test.py"

echo -e "${green}\n\nBuilding docker-image...${reset}"
docker build -t $name_image .

echo -e "${green}\n\nRemoving additional <none> images...${reset}"
docker rm $(docker ps -a -q) > /dev/null 2>&1
docker image prune -f

echo -e "${green}\n\nShow all images:${reset}"
docker image ls

echo -e "${green}\n\nRun docker-image:${reset}"
args="$@"
docker run \
-it --rm \
--gpus all \
--shm-size 8G \
--name $name_container \
--mount type=bind,source=/home/Korte/IDDATDLOCT/data,target=/IDDATDLOCT/data \
-i $name_image "src/${filename} ${args}"


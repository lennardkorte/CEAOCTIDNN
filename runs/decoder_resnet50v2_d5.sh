#! /bin/bash

entrance_script="./exec_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"

bash $entrance_script -cfg ./runs/decoder_resnet50v2_d5.json -wb $WANDB_API_KEY


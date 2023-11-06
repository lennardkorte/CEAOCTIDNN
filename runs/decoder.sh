#! /bin/bash

entrance_script="exe/run_train_and_eval_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"

bash $entrance_script -cfg ./runs/decoder.json -wb $WANDB_API_KEY -ycf -smp

# docker rmi -f $(docker images -aq)

# bash ./runs/decoder_gridsearch.sh

# sudo rm -r

# sudo apt install nvidia-driver-535 nvidia-dkms-535
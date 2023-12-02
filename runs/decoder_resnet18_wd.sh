#! /bin/bash

entrance_script="./exec_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"

for (( c=128; c<=1000; c=c*2 ))
do  
    bash $entrance_script -w $WANDB_API_KEY -cfg ./runs/decoder_resnet18_wd.json --wd $((c))e-7 --nm "run_wd$((c))e-7"
done
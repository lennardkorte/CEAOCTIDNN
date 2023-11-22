#! /bin/bash

entrance_script="exe/run_train_and_eval_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"

for (( b=16; b<=128; b=c*2 ))
do
    for (( c=1; c<=1000; c=c*2 ))
    do  
        bash $entrance_script -w $WANDB_API_KEY -cfg ./runs/decoder_resnet18_bs$((b)).json --lr $((c))e-4 --nm "run_bs$((b))_$((c))e-4" --bs $((b))
    done
done
#! /bin/bash

entrance_script="exe/run_train_and_eval_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"

for (( c=1; c<=10000; c=c*2 ))
do  
    bash $entrance_script -w $WANDB_API_KEY -cfg ./tests/01_config_lr.json --lr $((c))e-8 --nm "run$((c))e-8"
done


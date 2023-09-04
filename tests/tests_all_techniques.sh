#! /bin/bash

entrance_script="exe/run_train_and_eval_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"


for (( c=0; c<=10; c++ ))
do  
   bash $entrance_script -w $WANDB_API_KEY -cfg ./config_test.json --nm "run_da_$((c))"
done

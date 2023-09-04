#! /bin/bash

entrance_script="exe/run_train_and_eval_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"

bash $entrance_script -w $WANDB_API_KEY -cfg ./tests/02_tests_init.json --nm "std_0_padding2"



#! /bin/bash

entrance_script="exe/run_train_and_eval_docker.sh"
WANDB_API_KEY="8bea2fc48ca4a501eaec31dbfb410413a640b839"

for (( c=1; c<=1000; c=c*2 ))
do  
    bash $entrance_script -w $WANDB_API_KEY -cfg ./runs/decoder_bsOpt_wdxx.json --wd $((c))e-7 --nm "run_bsOpt_wd$((c))e-7" -ycf
done
#! /bin/bash

green=`tput setaf 2`
reset=`tput sgr0`

filename="train_and_test.py"

echo -e "${green}\n\nRun training...:${reset}"
args="$@"
python3 src/main.py "src/${filename} ${args}"
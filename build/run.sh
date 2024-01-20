#!/bin/bash
jupyter lab --allow-root --ip=* --no-browser --port 8888 --NotebookApp.token='tensorflow'
echo "alias ll="ls -al" > ~/.bashrc
source ~/.bashrc
/bin/bash

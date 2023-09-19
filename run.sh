#!/bin/bash
cd /projects
jupyter lab --allow-root --ip=* --no-browser --NotebookApp.token='tensorflow'
jupyter notebook --allow-root --ip=* --no-browser --NotebookApp.token='tensorflow'  --port 8080
while true; do sleep 1000; done
echo "alias ll="ls -al" > ~/.bashrc
/bin/bash
cd /projects

#!/bin/bash -v

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

nohup jupyter notebook >> logs/jupyter.log & echo $! >> $DIR/notebook.pid


#!/bin/bash -v

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
while read line
do
    kill -SIGTERM $line
    sleep 1
    kill -SIGKILL $line
done < $DIR/notebook.pid

cat '' > $DIR/notebook.pid


#!/bin/bash -v

NB_HOME=$HOME/.jupyter
NB_CONFIG=$NB_HOME/jupyter_notebook_config.py
NB_SHA=$(python -c "from notebook.auth import passwd; print passwd()")

echo "c.NotebookApp.password = '$NB_SHA'" >> $NB_CONFIG

openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout $NB_HOME/mykey.key -out $NB_HOME/mycert.pem

echo "c.NotebookApp.certfile = '$NB_HOME/mycert.pem'" >> $NB_CONFIG 
echo "c.NotebookApp.keyfile = '$NB_HOME/mykey.key'"   >> $NB_CONFIG 
echo "c.NotebookApp.ip = '*'"                         >> $NB_CONFIG 
echo "c.NotebookApp.open_browser = False"             >> $NB_CONFIG 
echo "c.NotebookApp.port = 8888"                      >> $NB_CONFIG 

mkdir logs

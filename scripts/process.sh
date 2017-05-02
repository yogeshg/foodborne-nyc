#!/bin/bash -x

for f in *.tar; do p=${f%.*}; gtar --xform="s/.*\/\(.*\)/${p}_\1/" -xvf $f; done
mkdir archive
mv *.tar archive/
egrep "Trainable" *summary.txt | sed "s/\([0-9]*\)_summary.*: \([0-9,]*\)/\1\|\2/" > models.psv
python models2html.py models.psv > view.html



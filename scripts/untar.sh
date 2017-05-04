#!/bin/bash -x

for f in *.tar; do p=${f%.*}; gtar --xform="s/.*\/\(.*\)/${p}_\1/" -xvf $f; done
mkdir archive
mv *.tar archive/

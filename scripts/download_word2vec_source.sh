#!/bin/bash -v
wget -P archive/ https://storage.googleapis.com/google-code-archive-source/v2/code.google.com/word2vec/source-archive.zip
unzip archive/source-archive.zip
echo "*" > word2vec/.gitignore

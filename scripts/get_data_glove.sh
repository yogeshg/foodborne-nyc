#!/bin/bash

cd data
wget http://nlp.stanford.edu/data/glove.840B.300d.zip
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip

unzip glove.840B.300d.zip
unzip glove.twitter.27B.zip


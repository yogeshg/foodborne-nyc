#!/bin/bash -v
date
time ./word2vec \
        -train /tmp/yo/foodborne/yelp_preprocessed_sample.txt \
        -initmodel ./vectors.bin \
        -output vectors_yelp_sample.txt \
        -save-vocab vocab_yelp_sample.txt \
        -cbow 1 -size 200 -window 8 \
        -negative 25 -hs 0 -sample 1e-4 \
        -threads 12 -binary 0 -iter 15
date

#!/bin/bash -v
date
time ./word2vec \
        -train /tmp/yo/foodborne/yelp_preprocessed.txt \
        -initmodel /tmp/yo/foodborne/GoogleNews-vectors-negative300.bin \
        -output /tmp/yo/foodborne/vectors_yelp.txt \
        -save-vocab /tmp/yo/foodborne/vocab_yelp.txt \
        -cbow 1 -size 300 -window 8 \
        -negative 25 -hs 0 -sample 1e-4 \
        -threads 20 -binary 0 -iter 15
date

import torch
from torch.autograd import Variable

import models
import main
import os
from collections import Counter
from contextlib import contextmanager

import numpy as np

# X = Variable(torch.cuda.LongTensor(main.training_set.X[:64]))

model_dirname = 'data/models/20171217_022340_811949/'

if main.training_set is None:
    main.load_data('twitter.biased', 'data/twitter_data/', 'data/glove.twitter.27B.200d.txt')


def indices2ngram(indices):
    return "_".join(map(main.indexer.get_token, indices))


def indices2words(indices):
    return map(main.indexer.get_token, indices)


def indices2text(indices):
    return " ".join(indices2words(indices))



HTML_START = """
<!DOCTYPE html>
<html>
<head>
    <style type="text/css">
        PAD {padding-left:5px; border:1px dotted #f8f8f8f8;}
        UNK {padding-left:15px; border:1px dotted #aaa;}
        <!--PAD {padding-left:5px;}-->
        <!--UNK {padding-left:15px;}-->
    </style>
</head>
<body>
"""
HTML_END = """
</body>
</html>
"""

@contextmanager
def open_html_doc(fname):
    with open(fname, 'w') as f:
        f.write(HTML_START)
        try:
            yield f
        finally:
            f.write(HTML_END)

def get_highlighted_word(text, r=0, b=0, alpha=0.5, mode='red-over-blue'):
    assert 0 <= b <= 1 and 0 <= r <= 1, 'b,r: {}, {}'.format(b, r)
    b *= alpha
    r *= alpha
    if mode == 'red-over-blue':
        html_format = \
            '<span \nstyle="background-color: rgba(0, 0, 255, {b});">'\
            '<span \nstyle="background-color: rgba(255, 0, 0, {r});">'\
            '{text}'\
            '</span>'\
            '</span>'.format
    elif mode == 'background-color':
        html_format = lambda r,b,text : \
            '<span \nstyle="color: rgb(0, 0, 0); background-color:rgb({r}, {g}, {b});">'\
            '{text}'\
            '</span>'.format(r=255-b*125, g=255-(r+b)*125, b=255-r*125, text=text)

    return html_format(r=r, b=b, text=text)


def get_html(words, normalized_heatmap_pos, normalized_heatmap_neg):
    highlighted_words = []
    for i, w in enumerate(words):
        r = normalized_heatmap_pos[i]
        b = -normalized_heatmap_neg[i]
        highlighted_words.append(get_highlighted_word(w+" ", r=r, b=b, mode='background-color'))
    return "".join(highlighted_words)


def hedge(x, floor, ceil):
    if x < floor:
        return floor
    elif x > ceil:
        return ceil
    else:
        return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def normalize_heatmap(heatmap, magnitude, floor, ceil):
    return Counter({k: hedge((v/np.abs(magnitude)), floor, ceil) for k,v in heatmap.items()})

def normalize_heatmap_sigmoid(heatmap, floor, ceil):
    return Counter({k: (ceil-floor)*sigmoid(v)+floor for k,v in heatmap.items()})



# earlier code:
# normalized_heatmap = Counter({k: hedge((v/np.abs(logit)), -1, 1) for k,v in heatmap.items()})
# normalized_heatmap = Counter({k: make_neg1_to_1(v) for k,v in normalized_heatmap.items()})


with open_html_doc('a.html') as f:

    batch_size = 32
    net = torch.load(os.path.join(model_dirname, 'checkpoint.net.pt'), map_location=lambda storage,y: storage)

    for batch_id in range(0, np.ceil(main.validation_set.y.shape[0] / float(batch_size)).astype(int)):

        batch_start = batch_size * batch_id
        batch_end = batch_size * (batch_id+1)

        X0 = Variable(torch.LongTensor(main.validation_set.X[batch_start:batch_end]))

        X5, weights, bias, ngrams_interest = models.forward_inspect(net, X0, main.indexer)

        for idx in range(main.validation_set.y[batch_start:batch_end].shape[0]):
            X0_numpy = X0[idx].data.cpu().numpy()
            X5_numpy = X5[idx].data.cpu().numpy()

            logit = X5_numpy[0]
            proba = sigmoid(logit)
            proba_red = hedge(2*proba-1, 0, 1)
            proba_blue = -hedge(2*proba-1, -1, 0)

            sentence = indices2text(X0_numpy)
            heatmap_pos, heatmap_neg = models.get_heatmap(idx, weights, ngrams_interest)
            heatmap_pos = normalize_heatmap(heatmap_pos, logit, 0, 1)
            heatmap_neg = normalize_heatmap(heatmap_neg, logit, -1, 0)
            # heatmap_pos = normalize_heatmap_sigmoid(heatmap_pos, 0, 1)
            # heatmap_neg = normalize_heatmap_sigmoid(heatmap_neg, -1, 0)

            f.write(get_highlighted_word('{0:.2f}'.format(proba), r=proba_red, b=proba_blue))
            f.write(get_html(indices2words(X0_numpy), heatmap_pos, heatmap_neg))
            f.write("\n</br>\n")
            # for i, vocab_i in enumerate(X0_numpy):
            #     print((heatmap[i], main.indexer.get_token(vocab_i)))



import torch
from torch.autograd import Variable

import models
import main
import util as u

import os
from collections import Counter
from itertools import product
from contextlib import contextmanager

import numpy as np
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

from collections import defaultdict

from inspector import Inspector, get_heatmap

"""
Data related functions
"""

def indices2ngram(indices):
    return "_".join(map(main.indexer.get_token, indices))

def indices2words(indices):
    return map(main.indexer.get_token, indices)

def indices2text(indices):
    return " ".join(indices2words(indices))

"""
HTML related functions
"""


class HighlightedHtml:
    START = """
    <!DOCTYPE html>
    <html>
    <head>
        <style type="text/css">

            /*PAD {padding-left:5px; border:1px dotted #f8f8f8f8;}*/
            UNK {padding-left:15px; border:1px dotted #aaa;}
            samples { display: table; }
            sample { display: table-row; }
            confusion_category { display: table-cell; border: solid 1px;}
            true_probability { display: table-cell; border: solid 1px;}
            predicted_probability { display: table-cell; border: solid 1px;}
            highlighted_text { display: table-cell; border: solid 1px;}

        </style>
    </head>
    <body>
    <samples>
    """
    END = """
    </samples>
    </body>
    </html>
    """

    SAMPLE_FORMAT = """
    <sample>
        <confusion_category>{confusion_category}</confusion_category>
        <true_probability>{true_probability}</true_probability>
        <predicted_probability>{predicted_probability}</predicted_probability>
        <highlighted_text>{highlighted_text}</highlighted_text>
    </sample>
    """

    def get_highlighted_word(text, r=0, b=0, alpha=0.5):
        assert 0 <= b <= 1 and 0 <= r <= 1, 'b,r: {}, {}'.format(b, r)
        b *= alpha
        r *= alpha
        if r>0 or b>0:
            return '<span style="background-color:rgb({r}, {g}, {b});">{text}'\
            '</span>'.format(r=255-b*125, g=255-(r+b)*125, b=255-r*125, text=text)
        else:
            return text


class HighlightedLatex:
    START = """
    \\providecommand{\\formatsample}[4]{
        #1 pred: #3 & {\\tiny #4} \\\\
        \\midrule
    }
    \\providecommand{\\formatsampletable}[1]{
        \\begin{tabular}{p{0.1\\textwidth}|p{0.9\\textwidth}}
        Category & Highlighted Text \\\\
        \\toprule
        #1
    }

    \\formatsampletable{
    """
    END = """
    }
    """

    SAMPLE_FORMAT = """
    \\formatsample
        {{{confusion_category}}}
        {{{true_probability}}}
        {{{predicted_probability}}}
        {{{highlighted_text}}}
    """

    @staticmethod
    def get_highlighted_word(text, r=0, b=0, alpha=0.5):
        assert 0 <= b <= 1 and 0 <= r <= 1, 'b,r: {}, {}'.format(b, r)
        b *= alpha
        r *= alpha
        if r > 0 or b > 0:
            return '\\highlightedword{{{r}}}{{{g}}}{{{b}}}{{{text}}}'.format(
                r = 1 - b * 0.5,
                g = 1 - (r+b) * 0.5,
                b = 1 - r * 0.5,
                text = text
            )
        else:
            return text

    @staticmethod
    def get_highlighted_words(words, normalized_heatmap_pos, normalized_heatmap_neg):
        highlighted_words = []
        for i, w in enumerate(words):
            r = normalized_heatmap_pos[i]
            b = -normalized_heatmap_neg[i]
            highlighted_words.append(HighlightedLatex.get_highlighted_word(w+" ", r=r, b=b))
        return "".join(highlighted_words)


@contextmanager
def open_html_doc(fname, formatting_class):
    with open(fname, 'w') as f:
        f.write(formatting_class.START)
        try:
            yield f
        finally:
            f.write(formatting_class.END)

def get_highlighted_word_redoverblue(text, r=0, b=0, alpha=0.5):
    assert 0 <= b <= 1 and 0 <= r <= 1, 'b,r: {}, {}'.format(b, r)
    b *= alpha
    r *= alpha
    return \
            '<span style="background-color: rgba(0, 0, 255, {b});">'\
            '<span style="background-color: rgba(255, 0, 0, {r});">'\
            '{text}'\
            '</span>'\
            '</span>'.format(r=r, b=b, text=text)


"""
Math related functions
"""

def get_confusion_category(y_pred, y_true, threshold):

    y_true = y_true.astype(bool)
    y_pred = (y_pred > threshold).astype(bool)
    true_positive = y_pred & y_true
    false_positive = y_pred & (~y_true)
    false_negative = (~y_pred) & y_true
    true_negative = (~y_pred) & (~y_true)

    category = map(lambda x: "".join(x), zip(
    map(lambda x: 'true positive' if x else '', true_positive),
        map(lambda x: 'false positive' if x else '', false_positive),
        map(lambda x: 'false negative' if x else '', false_negative),
        map(lambda x: 'true negative' if x else '', true_negative)
    ))

    return category

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


def get_all_html(model_dirname, dataset, embeddings, indexer, batch_size):
    indices = np.argwhere(dataset.z == 0)
    indices = [x[0] for x in indices]
    # print((indices))
    logger.info("consider a sample of {} for htmls".format(len(indices)))
    # print(dataset.z)
    # print(dataset.w)
    num_batches = int(np.ceil(len(indices) / float(batch_size)).astype(int))
    net = torch.load(os.path.join(model_dirname, 'checkpoint.net.pt'), map_location=lambda storage, y: storage)
    net_inspector = Inspector(net, embeddings)
    categorywise_all_html = defaultdict(list)

    for batch_id in range(0, num_batches):
        u.log_frequently(5, batch_id, logger.debug, 'processing batch {}'.format(batch_id))
        _batch_start = batch_size * batch_id
        _batch_end = batch_size * (batch_id + 1)
        batch_indices = indices[_batch_start: _batch_end]

        # print(dataset.X[_batch_start:_batch_end])
        # print(dataset.X[batch_indices])
        X0 = Variable(torch.cuda.LongTensor(dataset.X[batch_indices]))

        X5, weights, bias, ngrams_interest = net_inspector.forward_inspect(X0, indexer)
        yp = F.sigmoid(X5)
        yp = yp.resize(yp.size()[0])
        y_pred = yp.data.cpu().numpy()
        y_true = dataset.y[batch_indices]
        confusion_categories = get_confusion_category(y_pred, y_true, 0.5)


        for idx in range(dataset.y[batch_indices].shape[0]):
            X0_numpy = X0[idx].data.cpu().numpy()
            X5_numpy = X5[idx].data.cpu().numpy()

            logit = X5_numpy[0]
            proba = y_pred[idx]
            proba_red = hedge(2 * proba - 1, 0, 1)
            proba_blue = -hedge(2 * proba - 1, -1, 0)

            heatmap_pos, heatmap_neg = get_heatmap(idx, weights, ngrams_interest)
            heatmap_pos = normalize_heatmap(heatmap_pos, logit, 0, 1)
            heatmap_neg = normalize_heatmap(heatmap_neg, logit, -1, 0)
            # heatmap_pos = normalize_heatmap_sigmoid(heatmap_pos, 0, 1)
            # heatmap_neg = normalize_heatmap_sigmoid(heatmap_neg, -1, 0)

            confusion_category = confusion_categories[idx]
            true_probability = HighlightedLatex.get_highlighted_word('{0:.2f}'.format(y_true[idx]), r=y_true[idx], b=0)
            predicted_probability = HighlightedLatex.get_highlighted_word('{0:.2f}'.format(proba), r=proba_red,
                                                                          b=proba_blue)
            highlighted_text = HighlightedLatex.get_highlighted_words(indices2words(X0_numpy), heatmap_pos, heatmap_neg)
            sample_xml = HighlightedLatex.SAMPLE_FORMAT.format(confusion_category=confusion_category,
                                                               true_probability=true_probability,
                                                               predicted_probability=predicted_probability,
                                                               highlighted_text=highlighted_text)
            categorywise_all_html[confusion_category].append((sample_xml, y_true[idx], proba))

    return categorywise_all_html

"""
Main Code
"""

sample_size = 7

selected_models = {
"twitter.gold"   : "data/models/20171217_020721_811949/",
# "twitter.silver" : "data/models/20171217_175028_811949/",
# "twitter.biased" : "data/models/20171217_022127_811949/",
"yelp.gold"      : "data/models/20171217_061943_811949/",
# "yelp.silver"    : "data/models/20171217_195647_811949/",
# "yelp.biased"    : "data/models/20171217_203244_811949/",
}


dataset_media = ('twitter', 'yelp')
dataset_regimes = ('gold',)

data_paths = ('data/twitter_data/', 'data/yelp_data/')
embeddings_paths = ('data/glove.twitter.27B.200d.txt', 'data/glove.840B.300d.txt')

inputs = list(product(zip(dataset_media, data_paths, embeddings_paths), dataset_regimes))
for (medium, data_path, embeddings_path), regime in inputs:
        main.load_data(medium + '.gold', data_path, embeddings_path)
        # assert main.indexer._index2tokens[0][:5] == '<PAD>'
        main.indexer._index2tokens[0] = ''
        # assert main.indexer._index2tokens[1][:5] == '<UNK>'
        main.indexer._index2tokens[1] = '\unk'
        try:
            idx = main.indexer._tokens2index['#']
            main.indexer._index2tokens[idx] = '\\#'
            idx = main.indexer._tokens2index['$']
            main.indexer._index2tokens[idx] = '\\$'
            idx = main.indexer._tokens2index['&']
            main.indexer._index2tokens[idx] = '\\&'
        except Exception as e:
            logger.info('error while trying to replace "#" by "\\#"')
            logger.exception(e)

        dataset = medium + '.' + regime
        model_dirname = selected_models[dataset]
        batch_size = 32

        categorywise_all_html = get_all_html(
            model_dirname, main.testing_set, main.embeddings_matrix, main.indexer, batch_size
        )

        print([(cat, len(info))for cat, info in categorywise_all_html.items()])

        def agreement(truth, pred):
            return (2*pred - 1) * (2*truth - 1)

        def sample_if_needed(l, sz):
            if len(l) > sz:
                return sorted(l, key=lambda x: agreement(x[1], x[2]))[:sz]
                # return np.random.choice(l, replace=False, size=sz)
            else:
                return l
        category_samples = {cat: sample_if_needed(htmls, sample_size)\
                                    for cat, htmls in categorywise_all_html.items()}

        fname = 'stats/highlights.{}.tex'.format(dataset)
        logging.info('writing to file: {}'.format(fname))
        with open_html_doc(fname, HighlightedLatex) as f:
            for cat, info in sorted(category_samples.items()):
                f.write("\n".join(map(lambda x:x[0], info)))

        # break
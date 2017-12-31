# Neural Networks

## Word Embeddings
Word embeddings is the method where we represent a word by a vector in a multi-dimensional space.

### Word2Vec and Glove vectors
The word2vec algorithm starts with words projected randomly in a few hundred dimensional space by Gaussian distribution. Then the words that co-occur in the training set are moved closer together. After a few hundred iterations, the embeddings accumulate semantic meaning [cite]. This is critical for our algorithm as we expect that the words that represent similar meaning as, let's say `food poisoning` will have vectors close to each other.

For this project, we use two sets of Glove vectors that are provided by [Stanford NLP] Lab. These are trained on corpora collected common crawl on the Internet and Twitter respectively and are 300 and 200 dimensional.

### Unseen words
Embedding the words in a multi-dimensional space with a semantic structure takes care of unseen words since we can reasonably expect any unseen word to be projected close to a word vector that the net has seen before. If the seen word vector is important, the dot product of the convolution filter in question with the new vector will be high. Higher, the closer the new vector is.

## Convolution Neural Layers
Neural Networks have many layers that feed their outputs into the next layer. Output of each node of a fully connected layer is calculated by considering each point of the input. But in a convolution layer, the output of a node depends only on the local neighborhood of that node. 

#### FIXME
An `n`-dimensional convolution filter on an `m`-dimensional input will have `m-n`-dimensional parameters. In our case, we use 1-dimensional convolution filters, on a 2-dimensional input, and thus we have a 1 dimensional parameter. In image recognition, we have 2-dimensional features

### Soft Ngrams as convolutions
This is similar in spirit to n-grams where the impact of a word depends upon its neighbors. Mathematically, the activation of a 1-dimensional convolution of a certain filter size, say 3, is the dot product of that filter with the word and its neighbor hood of the same size, in this case, 1 to the left and 1 to the right.

#### TODO
Add image of a trigram and a convolution that picks it in a 2D space by PCA.

### Sparsity, larger n-grams
Typical vocabularies of languages are about a few tens of millions big and adding entities, proper nouns and misspellings can make this to a few hundreds of millions [verify]. The typical method of encoding as bag of words leads to each word being represented in a very sparse space. Because of embeddings, the vector space becomes much more dense and it becomes easier to train much more convolution filters in contrast to n-grams in case of bag-of-words.

## Regularization

Regularization is import to decrease the generalization error of neural nets. A lot of different techniques can be used for regularization, we describe the ones we use in our project in the following sections.

#### TODO
Look up the deep learning book for definitions of Regularization, Dropout, Weight Decay, Early Stopping.

### Dropout
Dropout can be done by dropping a node while training with a certain probability. We drop nodes from convolution filters randomly with a probability of `0.5`.
This method is viewed by many as training an ensemble of networks and using them all at the run time to make a decision.

### Weight Decay
#### TODO

### Early Stopping
We train the network for up to 100 epochs and stop if we have not seen an improvement in f1 score on the validation set for 25 epochs. Early stopping mechanism allows us to prevent the model from over fitting to the training data.

[verify]: (verification required)
[cite]: (citation required)
[Stanford NLP]: (check glove provider)
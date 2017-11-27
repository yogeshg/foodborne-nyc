import main
from itertools import product

dataset_media = ('twitter', 'yelp')
dataset_regimes = ('gold', 'silver', 'biased')

indexpath = 'data/vocab_yelp_sample.txt'
embeddingspath  = 'data/vectors_yelp_sample.txt'
for (media, regime) in product(dataset_media, dataset_regimes):
    dataset = media + '.' + regime
    main.load_data(dataset, indexpath, embeddingspath)
    other_params = {'embeddingspath':embeddingspath, 'dataset':dataset}
    main.run_experiments(finetune=False, kernel_sizes=(1,2,3,4),
                     filters=50, lr=1e-3, pooling='logsumexp',
                     kernel_l2_regularization=0.001, other_params=other_params)
    break


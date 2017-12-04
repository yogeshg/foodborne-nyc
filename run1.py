import main
from itertools import product

dataset_media = ('twitter', 'yelp')
dataset_regimes = ('gold', 'silver', 'biased')

indexpath = 'data/vocab_yelp.txt'
embeddingspath  = 'data/vectors_yelp.txt'
for (media, regime) in product(dataset_media, dataset_regimes):
    dataset = media + '.' + regime
    main.load_data(dataset, indexpath, embeddingspath)
    other_params = {'embeddingspath': embeddingspath, 'dataset': dataset}
    for finetune in (True, False):
        main.run_experiments(finetune=finetune, kernel_sizes=(1,2,3),
                         filters=50, lr=1e-3, pooling='max',
                         weight_decay=0.001, other_params=other_params)
    for kernel_sizes in ((1,2), (1,2,3,4)):
        main.run_experiments(finetune=False, kernel_sizes=kernel_sizes,
                         filters=50, lr=1e-3, pooling='max',
                         weight_decay=0.001, other_params=other_params)
    for filters in (5,10,20,25,50,75,100):
        main.run_experiments(finetune=False, kernel_sizes=(1,2,3),
                         filters=filters, lr=1e-3, pooling='max',
                         weight_decay=0.001, other_params=other_params)
    for pooling in ('max', 'logsumexp', 'average'):
        main.run_experiments(finetune=False, kernel_sizes=(1,2,3),
                         filters=50, lr=1e-3, pooling=pooling,
                         weight_decay=0.001, other_params=other_params)

import main
from itertools import product
import logging
logger = logging.getLogger(__name__)

dataset_media = ('twitter', 'yelp')
dataset_regimes = ('gold', 'silver', 'biased')

indexpath = 'data/vocab_yelp.txt'
embeddingspath  = 'data/vectors_yelp.txt'
experiment_id = 0
for (media, regime) in product(dataset_media, dataset_regimes):
    dataset = media + '.' + regime
    main.load_data(dataset, indexpath, embeddingspath)
    other_params = {'embeddingspath': embeddingspath, 'dataset': dataset}
    for finetune in (True, False):
        logging.info('running experiment_id: {}'.format(experiment_id))
        experiment_id+=1
        main.run_experiments(finetune=finetune, kernel_sizes=(1,2,3),
                         filters=50, lr=1e-3, pooling='max',
                         weight_decay=0.001, other_params=other_params)
    for kernel_sizes in ((1,2), (1,2,3,4)):
        logging.info('running experiment_id: {}'.format(experiment_id))
        experiment_id+=1
        main.run_experiments(finetune=False, kernel_sizes=kernel_sizes,
                         filters=50, lr=1e-3, pooling='max',
                         weight_decay=0.001, other_params=other_params)
    for filters in (5,10,20,25,50,75,100):
        logging.info('running experiment_id: {}'.format(experiment_id))
        experiment_id+=1
        main.run_experiments(finetune=False, kernel_sizes=(1,2,3),
                         filters=filters, lr=1e-3, pooling='max',
                         weight_decay=0.001, other_params=other_params)
    for pooling in ('max', 'logsumexp', 'average'):
        logging.info('running experiment_id: {}'.format(experiment_id))
        experiment_id+=1
        main.run_experiments(finetune=False, kernel_sizes=(1,2,3),
                         filters=50, lr=1e-3, pooling=pooling,
                         weight_decay=0.001, other_params=other_params)

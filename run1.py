import main
from itertools import product
import logging
logger = logging.getLogger(__name__)

dataset_media = ('twitter', 'yelp')
dataset_regimes = ('silver',)

data_paths = ('data/twitter_data/', 'data/yelp_data/')
embeddings_paths = ('data/glove.twitter.27B.200d.txt', 'data/glove.840B.300d.txt')
experiment_id = 0

inputs = list(product(zip(dataset_media, data_paths, embeddings_paths), dataset_regimes))
for hyperparameter_slice in (slice(None, -4), slice(-4, None)):
    for (medium, data_path, embeddings_path), regime in inputs:
        try:
            dataset = medium + '.' + regime
            main.load_data(dataset, data_path, embeddings_path)
            other_params = {'embeddings_path': embeddings_path, 'dataset': dataset}

            kernel_sizes_choices = ((1,2), (1,2,3), (1,2,3,4), (1,2,3,4,5))
            filters_choices = (5, 10, 20, 25, 50, 75, 100)
            num_params = lambda h: sum(h[0]) * h[1]

            hyperparameter_choices = sorted(product(kernel_sizes_choices, filters_choices), key=num_params)

            for kernel_sizes, filters in hyperparameter_choices[hyperparameter_slice]:
                logging.info('running experiment_id: {}'.format(experiment_id))
                experiment_id+=1
                try:
                    main.run_experiments(finetune=False, kernel_sizes=kernel_sizes,
                                     filters=filters, lr=1e-3, pooling='max',
                                     weight_decay=0.001, other_params=other_params)
                except Exception, e:
                    logger.exception(e)
        except Exception, e:
            logger.exception(e)


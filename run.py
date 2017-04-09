from models import *

def main():
    experiment_id = 0
    experiments_to_run = map(int, sys.argv[1:])
    indexpath = '/tmp/yo/foodborne/vocab_yelp.txt'
    embeddingspath  = '/tmp/yo/foodborne/vectors_yelp.txt'
    load_data('None', indexpath, embeddingspath, testdata=False)
    try:
      for nb_filter in (5,):
        lr = 1e-3
        pooling = 'max'
        filter_lengths_size = 3
        filter_lengths = tuple((x+1 for x in range(filter_lengths_size)))
        for kernel_l2_regularization in [0.01,]:
            if(experiment_id in experiments_to_run):
                logging.info('running experiment_id: {}'.format(experiment_id))
                run_experiments(finetune=False, filter_lengths=filter_lengths,
                    nb_filter=nb_filter, lr=lr, pooling=pooling, kernel_l2_regularization=kernel_l2_regularization,
                    other_params={'embeddingspath':embeddingspath})
            experiment_id += 1
    except Exception as e:
        logging.exception(e)
    return

if __name__ == '__main__':
    main()


from models import *

def main():
    experiment_id = 0
    experiments_to_run = map(int, sys.argv[1:])
    datapath = '/tmp/yo/foodborne/yelp_labelled.csv'
    indexpath = '/tmp/yo/foodborne/vocab_yelp.txt'
    embeddingspath  = '/tmp/yo/foodborne/vectors_yelp.txt'
    load_data(datapath, indexpath, embeddingspath, testdata=False)
    try:
      for filters in (5,10,25,50):
        lr = 1e-3
        kernel_l2_regularization = 0.0
        kernel_sizes_size = 3
        kernel_sizes = tuple((x+1 for x in range(kernel_sizes_size)))
        for pooling in ['max', 'avg', 'logsumexp']:
            if(experiment_id in experiments_to_run):
                logging.info('running experiment_id: {}'.format(experiment_id))
                run_experiments(finetune=False, kernel_sizes=kernel_sizes,
                    filters=filters, lr=lr, pooling=pooling, kernel_l2_regularization=kernel_l2_regularization,
                    other_params={'embeddingspath':embeddingspath})
            experiment_id += 1
    except Exception as e:
        logging.exception(e)
    return

if __name__ == '__main__':
    main()


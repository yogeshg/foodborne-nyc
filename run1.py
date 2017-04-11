from models import *

def main():
    datapath = '/tmp/yo/foodborne/yelp_labelled.csv'
    indexpath = '/tmp/yo/foodborne/vocab_yelp.txt'
    embeddingspath  = '/tmp/yo/foodborne/vectors_yelp.txt'
    load_data(datapath, indexpath, embeddingspath, testdata=False)
    other_params = {'embeddingspath':embeddingspath}
    run_experiments(finetune=False, kernel_sizes=(1,2,3), filters=5, lr=1e-3, pooling='logsumexp', kernel_l2_regularization=0.01, other_params=other_params)
    run_experiments(finetune=False, kernel_sizes=(1,2,3), filters=5, lr=1e-3, pooling='logsumexp', kernel_l2_regularization=0.001, other_params=other_params)
    run_experiments(finetune=False, kernel_sizes=(1,2,3), filters=5, lr=1e-3, pooling='logsumexp', kernel_l2_regularization=0.0001, other_params=other_params)

    run_experiments(finetune=False, kernel_sizes=(1,2,3), filters=25, lr=1e-3, pooling='logsumexp', kernel_l2_regularization=0.01, other_params=other_params)
    run_experiments(finetune=False, kernel_sizes=(1,2,3), filters=25, lr=1e-3, pooling='logsumexp', kernel_l2_regularization=0.001, other_params=other_params)
    run_experiments(finetune=False, kernel_sizes=(1,2,3), filters=25, lr=1e-3, pooling='logsumexp', kernel_l2_regularization=0.0001, other_params=other_params)

if __name__ == '__main__':
    main()


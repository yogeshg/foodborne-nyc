import main
from itertools import product
def run():
    dataset_media = ('twitter', 'yelp')
    dataset_regimes = ('gold', 'silver', 'biased')

    indexpath = 'data/vocab_yelp.txt'
    embeddingspath  = 'data/vectors_yelp.txt'
    for (media, regime) in product(dataset_media, dataset_regimes):
        dataset = media + '.' + regime
        main.load_data(dataset, indexpath, embeddingspath, testdata=False)
        other_params = {'embeddingspath':embeddingspath, 'dataset':dataset}
        main.run_experiments(finetune=False, kernel_sizes=(1,2,3,4),
                         filters=50, lr=1e-3, pooling='logsumexp',
                         kernel_l2_regularization=0.001, other_params=other_params)

if __name__ == '__main__':
    run()

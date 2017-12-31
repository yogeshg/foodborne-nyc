# Methodology

## The Model

Our model consists of 1-dimensional convolution filters that range from size 1 to 5, corresponding to unigrams, bigrams, trigrams, 4-grams and 5-grams.
We can vary this hyperparameter and call it `kernel_sizes`.
The number of filters we choose for each is also a hyper parameter that we call, `num_filters`.
We apply *Dropout* and *Max Pooling* layers to the output of these filters.
We pass the output of these through a *fully connected layer* to get the final output.
The number of parameters we have can be given by:

    def get_num_parameters(filter_sizes, num_filters):
      conv_params = sum(filter_sizes) * num_filters
      fc_params = len(filter_sizes) * num_filters
      return conv_params + fc_params

#### schematic of the model
TODO


## Hyperparameter Search
We search for the best hyperparameters in the `kernel_sizes x num_filters` space.

## Results Table
TODO

## Best Hyperparameters
#### Best Results
Because of weighted training methodology, sometimes the validation f1-scores that are very close to each other can be attribute to randomness. To select the best results, we take the top 5 models by validation f1-score and choose the one with the best validation recall.

#### Interpretation (of best results)
TODO


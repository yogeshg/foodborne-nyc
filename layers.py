from keras.layers import Conv1D, Dropout, concatenate

# layers
class LogSumExpPooling(Layer):

    def call(self, x):
        # could be axis 0 or 1
        return tf.reduce_logsumexp(x, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[:1]+input_shape[2:]

# layers
def get_conv_stack(input_layer, filters, kernel_sizes, activation, kernel_l2_regularization, dropout_rate):
    layers = [Conv1D(activation=activation, padding='same', strides=1, filters=filters, kernel_size = size,
                kernel_regularizer=regularizers.l2(kernel_l2_regularization))(input_layer) for size in kernel_sizes]
    if (len(layers) <= 0):
        return input_layer
    elif (len(layers) == 1):
        return Dropout(dropout_rate, noise_shape=None, seed=None)(layers[0])
    else:
        return Dropout(dropout_rate, noise_shape=None, seed=None)(concatenate(layers))

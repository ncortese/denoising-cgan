import tensorflow as tf


def build_conv_network(in_layer, layer_types, filter_shapes, strides, filter_nums, activation_func, reuse):
    """
    Builds a neural network with stacked convolution, transposed convolution (for generator) and pooling layers
    :param in_layer: input from tensorflow graph
    :param layer_types: tuple of layer types ('c': conv layer, 't': transposed conv layer, 'p': pooling layer) per layer
    :param filter_shapes: tuple of filter sizes per layer, of form (height, width)
    :param strides: tuple of stride sizes per layer, of form (height, width)
    :param filter_nums: tuple of number of filters per layer (can just put None for pooling layers)
    :param activation_func: tensorflow activation function
    :param reuse: Boolean, whether or not to reuse the weights of a previous layer by the same name
    :return: output to plug into tensorflow graph
    """

    # iterate through layers, composing each with previous layer
    depth = len(layer_types)
    layer = in_layer
    for i in range(depth):
        type_ = layer_types[i]
        if type_ == 'c':
            layer = tf.layers.conv2d(layer, filter_nums[i], filter_shapes[i], strides=strides[i],
                                     activation=activation_func,
                                     reuse=reuse,
                                     padding='same',
                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                     bias_initializer=tf.constant_initializer(0.1),
                                     name='conv%s' % i)
        elif type_ == 't':
            layer = tf.layers.conv2d_transpose(layer, filter_nums[i], filter_shapes[i], strides=strides[i],
                                               activation=activation_func,
                                               reuse=reuse,
                                               padding='same',
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                               bias_initializer=tf.constant_initializer(0.1),
                                               name='convtrans%s' % i)
        elif type_ == 'p':
            layer = tf.layers.max_pooling2d(layer, filter_shapes[i], strides[i], padding='same', name='pooling%s' % i)
        else:
            raise ValueError("only 'c', 't' and 'p' are valid layer types")
    return layer


def build_dense_network(in_layer, layer_types, layer_sizes, dropout_rates, activation_func, is_training, reuse):
    """
    Builds a neural network with fully connected layers
    :param in_layer: input from tensorflow graph
    :param layer_types: tuple of characters ('f': fully connected layer, 'd': dropout layer) per layer
    :param layer_sizes: tuple of number of units per layer (can just put None for dropout layers)
    :param dropout_rates: tuple of dropout rate per layer (can just put None for fully connected layers)
    :param activation_func: tensorflow activation function
    :param is_training: either boolean or tensorflow boolean placeholder (true: apply dropout, false: no dropout)
    :param reuse: Boolean, whether or not to reuse the weights of a previous layer by the same name
    :return: output to plug into tensorflow graph
    """

    # iterate through layers, composing each with previous layer
    depth = len(layer_sizes)
    layer = in_layer
    for i in range(depth):
        type_ = layer_types[i]
        if type_ == 'f':
            layer = tf.layers.dense(layer, layer_sizes[i], activation=activation_func,
                                    reuse=reuse,
                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                    bias_initializer=tf.constant_initializer(0.1),
                                    name='dense%s' % i)
        elif type_ == 'd':
            layer = tf.layers.dropout(layer, dropout_rates[i], training=is_training, name='dropout%s' % i)
        else:
            raise ValueError("only 'f' and 'd' are valid layer types")
    return layer

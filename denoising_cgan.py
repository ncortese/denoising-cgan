import tensorflow as tf
import nn_construction as nnc
import numpy as np

# debug
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# parameters
batch_size = 20
num_steps = 10000
image_height = 28
image_width = 28
image_channels = 1
image_size = image_height*image_width
activation = tf.nn.elu

# discriminator network architecture
disc_conv_layers = ['c']
disc_filter_shapes = [[4, 4]]
disc_strides = [[1, 1]]
disc_filter_nums = [24]
disc_dense_layers = ['f']
disc_dense_sizes = [1]
disc_dropout_rates = [None]

# generator network architecture
gen_conv_layers = []
gen_filter_shapes = []
gen_strides = []
gen_filter_nums = []
gen_dense_layers = ['f', 'f']
gen_dense_sizes = [128, 784]
gen_dropout_rates = [None, None]
latent_size = 32  # dimensionality of the noise for the generator
noise_type = tf.random_normal  # distribution of the noise for the generator


# build discriminator, following convolutional layers with dense layers
def discriminator(in_layer, reuse):
    out_layer = in_layer
    if len(disc_conv_layers) > 0:
        out_layer = tf.reshape(out_layer, [batch_size, image_height, image_width, -1])
        out_layer = nnc.build_conv_network(out_layer, disc_conv_layers, disc_filter_shapes, disc_strides,
                                           disc_filter_nums, activation, reuse)
    out_layer = tf.reshape(out_layer, [batch_size, -1])
    if len(disc_dense_layers) > 0:
        out_layer = nnc.build_dense_network(out_layer, disc_dense_layers, disc_dense_sizes,
                                            disc_dropout_rates, activation, False, reuse)
    return out_layer


# build generator, following dense layers with convolutional layers
def generator(in_layer, reuse):
    out_layer = in_layer
    if len(gen_dense_layers) > 0:
        out_layer = nnc.build_dense_network(out_layer, gen_dense_layers, gen_dense_sizes,
                                            gen_dropout_rates, activation, False, reuse)
    if len(gen_conv_layers) > 0:
        # following convolutional GAN paper's recommendation to reshape to a small spatial extent with many channels
        out_layer = tf.reshape(out_layer, [batch_size, 4, 4, -1])
        out_layer = nnc.build_conv_network(out_layer, gen_conv_layers, gen_filter_shapes, gen_strides,
                                           gen_filter_nums, activation, reuse)
    return out_layer

training_input = tf.placeholder(tf.float32, [batch_size, image_size])
cond_input = tf.placeholder(tf.float32, [batch_size, image_size])
gen_noise = noise_type([batch_size, latent_size])

# get outputs of generator
gen_input = tf.concat([gen_noise, cond_input], 1)
with tf.variable_scope('gen'):
    gen_output = tf.nn.sigmoid(generator(gen_input, False))

# get outputs of discriminator
disc_train_input = tf.concat([training_input, cond_input], 1)
disc_gen_input = tf.concat([gen_output, cond_input], 1)
with tf.variable_scope('disc'):
    disc_train_output = tf.nn.sigmoid(discriminator(disc_train_input, False))
    disc_gen_output = tf.nn.sigmoid(discriminator(disc_gen_input, True))

disc_loss = -tf.reduce_mean(tf.log(disc_train_output) + tf.log(1. - disc_gen_output))
gen_loss = -tf.reduce_mean(tf.log(disc_gen_output))

# to estimate progress
disc_train_accuracy = tf.reduce_mean(disc_train_output)
disc_gen_accuracy = tf.reduce_mean(disc_gen_output)

# get lists of discriminator and generator variables
disc_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc')
gen_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')

# disc_minimize = tf.train.AdamOptimizer().minimize(disc_loss, var_list=disc_variables)
# gen_minimize = tf.train.AdamOptimizer().minimize(gen_loss, var_list=gen_variables)

# clipped optimization for discriminator
with tf.name_scope("Adam_optimizer_disc"):
    optimizer = tf.train.AdamOptimizer(1E-4)
    grads_and_vars = optimizer.compute_gradients(disc_loss, var_list=disc_variables)
    clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
               for grad, tvar in grads_and_vars]
    train_step_disc = optimizer.apply_gradients(clipped, name="minimize_cost")

# clipped optimization for generator
with tf.name_scope("Adam_optimizer_gen"):
    optimizer = tf.train.AdamOptimizer(1E-4)
    grads_and_vars = optimizer.compute_gradients(gen_loss, var_list=gen_variables)
    clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
               for grad, tvar in grads_and_vars]
    train_step_gen = optimizer.apply_gradients(clipped, name="minimize_cost")

# Running on MNIST at first, for testing
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        batch, _ = mnist.train.next_batch(batch_size)

        # Sloppy temporary way of adding (masking) noise to each MNIST sample consistently (will remove this after building
        # actual dataset of 8x8 patches)
        noisy_batch = np.zeros([batch_size, image_size])
        for j in range(batch_size):
            seed_j = np.sum(batch[j, :]).astype(np.int64)
            np.random.seed(seed_j)
            noise = np.random.randint(2, size=image_size)
            noisy_batch[j, :] = np.multiply(batch[j, :], noise)

        # disc_minimize.run(feed_dict={training_input: batch, cond_input: noisy_batch})
        # gen_minimize.run(feed_dict={cond_input: noisy_batch})
        sess.run(train_step_disc, feed_dict={training_input: batch, cond_input: noisy_batch})
        sess.run(train_step_gen, feed_dict={cond_input: noisy_batch})
        if i % 100 == 0:
            current_disc_accuracy = disc_train_accuracy.eval(feed_dict={training_input: batch, cond_input: noisy_batch})
            current_gen_accuracy = disc_gen_accuracy.eval(feed_dict={training_input: batch, cond_input: noisy_batch})
            print('step {}, disc. train accuracy {:f}, disc. gen. accuracy {:f}'.format(i, current_disc_accuracy,
                                                                                        current_gen_accuracy))

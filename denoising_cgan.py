import tensorflow as tf
import nn_construction as nnc
import numpy as np
from batch_loader import BatchLoader
from PIL import Image
import os

# parameters
batch_size = 20
num_steps = 5000
image_height = 8
image_width = 8
image_channels = 3
image_size = image_height*image_width
activation = tf.nn.relu

# discriminator network architecture
disc_conv_layers = ['c', 'c', 'c', 'c']
disc_filter_shapes = [[4, 4], [4, 4], [4, 4], [1, 1]]
disc_strides = [[1, 1], [2, 2], [2, 2], [2, 2]]
disc_filter_nums = [8, 16, 16, 1]
disc_dense_layers = []
disc_dense_sizes = []
disc_dropout_rates = []

# generator network architecture
gen_conv_layers = ['t', 't', 't']
gen_filter_shapes = [[4, 4], [4, 4], [1, 1]]
gen_strides = [[1, 1], [2, 2], [1, 1]]
gen_filter_nums = [32, 16, 3]
gen_dense_layers = []
gen_dense_sizes = []
gen_dropout_rates = []
latent_size = 32  # dimensionality of the noise for the generator
noise_type = tf.random_uniform  # distribution of the noise for the generator


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
    out_layer = tf.reshape(out_layer, [batch_size, -1])
    return out_layer

training_input = tf.placeholder(tf.float32, [batch_size, image_size*image_channels])
cond_input = tf.placeholder(tf.float32, [batch_size, image_size*image_channels])
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

disc_minimize = tf.train.AdamOptimizer(learning_rate=2E-4, beta1=0.5).minimize(disc_loss, var_list=disc_variables)
gen_minimize = tf.train.AdamOptimizer(learning_rate=2E-4, beta1=0.5).minimize(gen_loss, var_list=gen_variables)

# clipped optimization for discriminator
# with tf.name_scope("Adam_optimizer_disc"):
#     optimizer = tf.train.AdamOptimizer(1E-4)
#     grads_and_vars = optimizer.compute_gradients(disc_loss, var_list=disc_variables)
#     clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
#                for grad, tvar in grads_and_vars]
#     train_step_disc = optimizer.apply_gradients(clipped, name="minimize_cost")
#
# # clipped optimization for generator
# with tf.name_scope("Adam_optimizer_gen"):
#     optimizer = tf.train.AdamOptimizer(1E-4)
#     grads_and_vars = optimizer.compute_gradients(gen_loss, var_list=gen_variables)
#     clipped = [(tf.clip_by_value(grad, -5, 5), tvar)  # gradient clipping
#                for grad, tvar in grads_and_vars]
#     train_step_gen = optimizer.apply_gradients(clipped, name="minimize_cost")

batches = BatchLoader('./berkeley_patches', './berkeley_noise')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(num_steps):
        batch_patches, batch_noise = batches.get_next_batch(batch_size)

        batch_patches = np.array(batch_patches)
        batch_noise = np.array(batch_noise)
        batch_patches = np.reshape(batch_patches, (batch_size, -1))
        batch_noise = np.reshape(batch_noise, (batch_size, -1))
        batch_patches = batch_patches.astype(np.float32)
        batch_noise = batch_noise.astype(np.float32)
        batch_patches = batch_patches/255.
        batch_noise = batch_noise/255.

        sess.run(disc_minimize, feed_dict={training_input: batch_patches, cond_input: batch_noise})
        sess.run(gen_minimize, feed_dict={cond_input: batch_noise})
        if i % 100 == 0:
            current_disc_accuracy = disc_train_accuracy.eval(feed_dict={training_input: batch_patches,
                                                                        cond_input: batch_noise})
            current_gen_accuracy = disc_gen_accuracy.eval(feed_dict={training_input: batch_patches,
                                                                     cond_input: batch_noise})
            print('step {}, disc. train accuracy {:f}, disc. gen. accuracy {:f}'.format(i, current_disc_accuracy,
                                                                                        current_gen_accuracy))
            # save a generated sample
            samples = gen_output.eval(feed_dict = {cond_input: batch_noise})
            samples = np.reshape(samples, (batch_size, image_height, image_width, image_channels))
            noise_1 = np.reshape(batch_noise[0, :], (image_height, image_width, image_channels))
            data_1 = np.reshape(batch_patches[0, :], (image_height, image_width, image_channels))
            sample_1 = samples[0, :, :, :]
            image = np.concatenate((data_1, noise_1, sample_1), axis=1)
            image = Image.fromarray(np.uint8(image*255))
            image.save(os.path.join('./generated_samples', "iteration_" + str(i) + ".jpg"), "JPEG")

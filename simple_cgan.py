import tensorflow as tf
import numpy as np
from batch_loader import BatchLoader
from PIL import Image
import os


def leaky_relu(x):
    return tf.maximum(x, leaky_relu_alpha*x)

# parameters
batch_size = 64
num_steps = 20000
image_height = 8
image_width = 8
image_channels = 3
image_size = image_height*image_width

latent_size = 32  # dimensionality of the noise for the generator
noise_type = tf.random_uniform  # distribution of the noise for the generator

leaky_relu_alpha = 0.2

training_input = tf.placeholder(tf.float32, [batch_size, image_height, image_width, image_channels])
cond_input = tf.placeholder(tf.float32, [batch_size, image_height, image_width, image_channels])

# get outputs of generator
gen_noise = noise_type([batch_size, latent_size])
gen_input = tf.concat([gen_noise, tf.reshape(cond_input, [batch_size, -1])], 1)
with tf.variable_scope('gen'):
    # map input linearly and reshape to 4x4x32
    # for some reason running xavier initializer gives errors. I switched to truncated normal for now since I didn't
    # feel like looking through tensorflow source code to try and fix the problem
    gen_1_matmul = tf.layers.dense(gen_input, 4*4*32, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    gen_1_matmul = tf.reshape(gen_1_matmul, [batch_size, 4, 4, -1])
    gen_2_convt = tf.layers.conv2d_transpose(gen_1_matmul, filters=24, kernel_size=(4, 4), padding='same',
                                             activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    gen_3_convt = tf.layers.conv2d_transpose(gen_2_convt, filters=16, kernel_size=(4, 4), strides=(2, 2),
                                             padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    gen_4_convt = tf.layers.conv2d_transpose(gen_3_convt, filters=8, kernel_size=(4, 4), padding='same',
                                             activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    gen_output = tf.layers.conv2d_transpose(gen_4_convt, filters=3, kernel_size=(1, 1), padding='same',
                                            activation=lambda x: 0.5*tf.nn.tanh(x) + 0.5, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))

# get outputs of discriminator
disc_train_input = tf.concat([training_input, cond_input], 1)
disc_gen_input = tf.concat([gen_output, cond_input], 1)
# disc_gen_input = tf.Print(disc_gen_input, [tf.shape(disc_gen_input)])
# disc_train_input = tf.Print(disc_train_input, [tf.shape(disc_train_input)])
with tf.variable_scope('disc'):
    # output on training data
    disc_1_train_conv = tf.layers.conv2d(disc_train_input, filters=32, kernel_size=(4, 4), padding='same',
                                         name="conv1", activation=leaky_relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    disc_2_train_conv = tf.layers.conv2d(disc_1_train_conv, filters=24, kernel_size=(4, 4), strides=(2, 2),
                                         name="conv2", padding='same', activation=leaky_relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    disc_3_train_conv = tf.layers.conv2d(disc_2_train_conv, filters=16, kernel_size=(4, 4), padding='same',
                                         name="conv3", activation=leaky_relu, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    disc_train_output = tf.layers.conv2d(disc_3_train_conv, filters=3, kernel_size=(1, 1), padding='same',
                                         name="out", activation=tf.nn.sigmoid, kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))
    # output on generator
    disc_1_gen_conv = tf.layers.conv2d(disc_gen_input, filters=32, kernel_size=(4, 4), padding='same',
                                       name="conv1", activation=leaky_relu, reuse=True)
    disc_2_gen_conv = tf.layers.conv2d(disc_1_gen_conv, filters=24, kernel_size=(4, 4), strides=(2, 2),
                                       name="conv2", padding='same', activation=leaky_relu, reuse=True)
    disc_3_gen_conv = tf.layers.conv2d(disc_2_gen_conv, filters=16, kernel_size=(4, 4), padding='same',
                                       name="conv3", activation=leaky_relu, reuse=True)
    disc_gen_output = tf.layers.conv2d(disc_3_gen_conv, filters=3, kernel_size=(1, 1), padding='same',
                                       name="out", activation=tf.nn.sigmoid, reuse=True)

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
    saver = tf.train.Saver()
    saver_gen = tf.train.Saver(gen_variables)
    for i in range(num_steps):
        batch_patches, batch_noise = batches.get_next_batch(batch_size)

        batch_patches = np.array(batch_patches)
        batch_noise = np.array(batch_noise)
        batch_patches = batch_patches.astype(np.float32)
        batch_noise = batch_noise.astype(np.float32)
        batch_patches = batch_patches/255.
        batch_noise = batch_noise/255.

        sess.run(disc_minimize, feed_dict={training_input: batch_patches, cond_input: batch_noise})
        sess.run(gen_minimize, feed_dict={cond_input: batch_noise})
        if i % 1000 == 0:
            print("step %s" % i)
            current_disc_accuracy = disc_train_accuracy.eval(feed_dict={training_input: batch_patches,
                                                                        cond_input: batch_noise})
            current_gen_accuracy = disc_gen_accuracy.eval(feed_dict={training_input: batch_patches,
                                                                     cond_input: batch_noise})
            print('step {}, disc. train accuracy {:f}, disc. gen. accuracy {:f}'.format(i, current_disc_accuracy,
                                                                                        current_gen_accuracy))
            # save a generated sample
            samples = gen_output.eval(feed_dict = {cond_input: batch_noise})
            noise_1 = batch_noise[0, :, :, :]
            data_1 = batch_patches[0, :, :, :]
            sample_1 = samples[0, :, :, :]
            image = np.concatenate((data_1, noise_1, sample_1), axis=1)
            image = np.maximum(image, 0)
            image = Image.fromarray(np.uint8(image*255))
            image.save(os.path.join('./generated_samples', "iteration_" + str(i) + ".jpg"), "JPEG")
        if i % 10000 == 0 and i != 0:
            save_path = saver.save(sess, "./saved_models/model", global_step=i)
            print("saving model to %s" % save_path)
    # save final generator separately for sampling
    save_path_gen = saver_gen.save(sess, "./saved_models/generator.ckpt")
    print("saving generator to %s" % save_path_gen)

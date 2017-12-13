import tensorflow as tf
import numpy as np
from PIL import Image
import os


# parameter to avoid NaN errors in loss
epsilon = 1e-10


class CGAN:
    def __init__(self, image_height, image_width, image_channels, noise_type, leaky_relu_alpha, disc_depth,
                 disc_filter_nums, disc_filter_sizes, disc_strides, gen_depth, gen_filter_nums,
                 gen_filter_sizes, gen_strides):
        self.image_height = image_height
        self.image_width = image_width
        self.image_channels = image_channels
        self.noise_type = noise_type

        def leaky_relu(x):
            return tf.maximum(x, leaky_relu_alpha * x)

        self.leaky_relu = leaky_relu
        self.disc_depth = disc_depth
        self.disc_filter_nums = disc_filter_nums
        self.disc_filter_sizes = disc_filter_sizes
        self.disc_strides = disc_strides
        self.gen_depth = gen_depth
        self.gen_filter_nums = gen_filter_nums
        self.gen_filter_sizes = gen_filter_sizes
        self.gen_strides = gen_strides

    def train(self, batch_loader, batch_size, num_steps, saved_checkpoint_path, saved_samples_path, restore_from_prev, restore_num=-1):
        training_input = tf.placeholder(tf.float32, [batch_size, self.image_height, self.image_width, self.image_channels])
        cond_input = tf.placeholder(tf.float32, [batch_size, self.image_height, self.image_width, self.image_channels])

        gen_noise = self.noise_type([batch_size, self.image_height, self.image_width, 1])
        gen_input = tf.concat([gen_noise, cond_input], 3)

        with tf.variable_scope('gen'):
            # iterate to compose generator layers
            gen_layer = gen_input
            for i in range(self.gen_depth):
                gen_layer = tf.layers.conv2d_transpose(gen_layer, filters=self.gen_filter_nums[i],
                                                       kernel_size=self.gen_filter_sizes[i],
                                                       strides=self.gen_strides[i], padding='same',
                                                       activation=tf.nn.relu,
                                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))

            gen_output = tf.layers.conv2d_transpose(gen_layer, filters=self.image_channels,
                                                    kernel_size=(1, 1), padding='same',
                                                    activation=lambda x: 0.5*tf.nn.tanh(x) + 0.5,
                                                    kernel_initializer=
                                                    tf.truncated_normal_initializer(stddev=0.2)) + cond_input

        # get outputs of discriminator
        disc_train_input = tf.concat([training_input, cond_input], 1)
        disc_gen_input = tf.concat([gen_output, cond_input], 1)
        with tf.variable_scope('disc'):
            # output on training data
            disc_train_layer = disc_train_input
            for i in range(self.disc_depth):
                disc_train_layer = tf.layers.conv2d(disc_train_layer, filters=self.disc_filter_nums[i],
                                                    kernel_size=self.disc_filter_sizes[i],
                                                    strides=self.disc_strides[i], padding='same',
                                                    activation=self.leaky_relu,
                                                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                                                    name="conv%s" % i)
            disc_train_output = tf.layers.conv2d(disc_train_layer, filters=1, kernel_size=(1, 1), padding='same',
                                                 name="out", activation=tf.nn.sigmoid,
                                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))

            # output on generator
            disc_gen_layer = disc_gen_input
            for i in range(self.disc_depth):
                disc_gen_layer = tf.layers.conv2d(disc_gen_layer, filters=self.disc_filter_nums[i],
                                                  kernel_size=self.disc_filter_sizes[i],
                                                  strides=self.disc_strides[i], padding='same',
                                                  activation=self.leaky_relu,
                                                  kernel_initializer=tf.truncated_normal_initializer(stddev=0.2),
                                                  name="conv%s" % i, reuse=True)
            disc_gen_output = tf.layers.conv2d(disc_gen_layer, filters=1, kernel_size=(1, 1), padding='same',
                                               name="out", activation=tf.nn.sigmoid, reuse=True,
                                               kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))

        disc_loss = -tf.reduce_mean(tf.log(disc_train_output + epsilon) + tf.log(1. - disc_gen_output + epsilon))
        gen_loss = -tf.reduce_mean(tf.log(disc_gen_output + epsilon))

        # to estimate progress
        disc_train_accuracy = tf.reduce_mean(disc_train_output)
        disc_gen_accuracy = tf.reduce_mean(disc_gen_output)

        # get lists of discriminator and generator variables
        disc_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='disc')
        gen_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gen')

        disc_minimize = tf.train.AdamOptimizer(learning_rate=2E-5, beta1=0.5).minimize(disc_loss,
                                                                                       var_list=disc_variables)
        gen_minimize = tf.train.AdamOptimizer(learning_rate=2E-5, beta1=0.5).minimize(gen_loss, var_list=gen_variables)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver_gen = tf.train.Saver(gen_variables)
            if restore_from_prev:
                saver.restore(sess, os.path.join(saved_checkpoint_path, "model-%s" % restore_num))
            for i in range(restore_num+1, num_steps+restore_num+1):
                batch_patches, batch_noise = batch_loader.get_next_batch(batch_size)

                batch_patches = np.array(batch_patches)
                batch_noise = np.array(batch_noise)
                batch_patches = batch_patches.astype(np.float32)
                batch_noise = batch_noise.astype(np.float32)
                batch_patches = batch_patches / 255.
                batch_noise = batch_noise / 255.

                sess.run(disc_minimize, feed_dict={training_input: batch_patches, cond_input: batch_noise})
                sess.run(gen_minimize, feed_dict={cond_input: batch_noise})
                if i % 1000 == 0:
                    print("step %s" % i)
                    current_disc_accuracy = disc_train_accuracy.eval(feed_dict={training_input: batch_patches,
                                                                                cond_input: batch_noise})
                    current_gen_accuracy = disc_gen_accuracy.eval(feed_dict={training_input: batch_patches,
                                                                             cond_input: batch_noise})
                    print(
                        'step {}, disc. train accuracy {:f}, disc. gen. accuracy {:f}'.format(i, current_disc_accuracy,
                                                                                              current_gen_accuracy))
                    # save a generated sample
                    samples = gen_output.eval(feed_dict={cond_input: batch_noise})
                    noise_1 = batch_noise[0, :, :, :]
                    data_1 = batch_patches[0, :, :, :]
                    sample_1 = samples[0, :, :, :]
                    image = np.concatenate((data_1, noise_1, sample_1), axis=1)
                    image = np.maximum(image, 0)
                    image = Image.fromarray(np.uint8(image * 255))
                    image.save(os.path.join(saved_samples_path, "iteration_" + str(i) + ".jpg"), "JPEG")
                if (i + 1) % 2000 == 0:
                    save_path = saver.save(sess, os.path.join(saved_checkpoint_path, "model"), global_step=i)
                    print("saving model to %s" % save_path)

                    # this is redundant with the above, but save it separately since that's how the image generator
                    # functions take it
                    save_path_gen = saver_gen.save(sess, os.path.join(saved_checkpoint_path, "generator"))
                    print("saving generator to %s" % save_path_gen)
            # save final generator
            save_path_gen = saver_gen.save(sess, os.path.join(saved_checkpoint_path, "generator"))
            print("saving generator to %s" % save_path_gen)

    def generate(self, tf_session, saved_generator_path):
        cond_input = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.image_channels])

        gen_noise = self.noise_type([1, self.image_height, self.image_width, 1])
        gen_input = tf.concat([gen_noise, cond_input], 3)

        with tf.variable_scope('gen'):
            # iterate to compose generator layers
            gen_layer = gen_input
            for i in range(self.gen_depth):
                gen_layer = tf.layers.conv2d_transpose(gen_layer, filters=self.gen_filter_nums[i],
                                                       kernel_size=self.gen_filter_sizes[i],
                                                       strides=self.gen_strides[i], padding='same',
                                                       activation=tf.nn.relu,
                                                       kernel_initializer=tf.truncated_normal_initializer(stddev=0.2))

            gen_output = tf.layers.conv2d_transpose(gen_layer, filters=self.image_channels,
                                                    kernel_size=(1, 1), padding='same',
                                                    activation=lambda x: 0.5*tf.nn.tanh(x) + 0.5,
                                                    kernel_initializer=
                                                    tf.truncated_normal_initializer(stddev=0.2)) + cond_input

        saver = tf.train.Saver()
        saver.restore(tf_session, saved_generator_path)
        return cond_input, gen_output

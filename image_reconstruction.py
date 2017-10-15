import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimage

noise_dev = 255./16.

image_filename = './berkeley data/train/65074.jpg'
im = Image.open(image_filename)

# crop for splitting into 8x8 patches
init_width, init_height = im.size
patches_w = init_width//8
patches_h = init_height//8
im_width = 8*patches_w
im_height = 8*patches_h
im = im.crop((0, 0, im_width, im_height))

array_im = mpimage.pil_to_array(im)
normal_noise = np.random.normal(scale=noise_dev, size=(im_height, im_width, 3))
array_noise = array_im + normal_noise
array_noise = (1/255)*array_noise

# build generator
image_height = 8
image_width = 8
image_channels = 3
image_size = image_height*image_width

latent_size = 32  # dimensionality of the noise for the generator
noise_type = tf.random_uniform  # distribution of the noise for the generator

# since tensors included batch size during training tensorflow gets mad if the first dimension is omitted
cond_input = tf.placeholder(tf.float32, [1, image_height, image_width, image_channels])

# get outputs of generator
gen_noise = noise_type([1, latent_size])
# gen_noise = tf.placeholder(tf.float32, [1, latent_size])
gen_input = tf.concat([gen_noise, tf.reshape(cond_input, [1, -1])], 1)
with tf.variable_scope('gen'):
    # map input linearly and reshape to 4x4x32
    # for some reason running xavier initializer gives errors. I switched to truncated normal for now since I didn't
    # feel like looking through tensorflow source code to try and fix the problem
    gen_1_matmul = tf.layers.dense(gen_input, 4*4*32, activation=None, kernel_initializer=tf.truncated_normal_initializer)
    gen_1_matmul = tf.reshape(gen_1_matmul, [1, 4, 4, -1])
    gen_2_convt = tf.layers.conv2d_transpose(gen_1_matmul, filters=24, kernel_size=(4, 4), padding='same',
                                             activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    gen_3_convt = tf.layers.conv2d_transpose(gen_2_convt, filters=16, kernel_size=(4, 4), strides=(2, 2),
                                             padding='same', activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    gen_4_convt = tf.layers.conv2d_transpose(gen_3_convt, filters=8, kernel_size=(4, 4), padding='same',
                                             activation=tf.nn.relu, kernel_initializer=tf.truncated_normal_initializer)
    gen_output = tf.layers.conv2d_transpose(gen_4_convt, filters=3, kernel_size=(1, 1), padding='same',
                                            activation=lambda x: 0.5*tf.nn.tanh(x) + 0.5, kernel_initializer=tf.truncated_normal_initializer)

reconstructed_image = np.zeros(array_noise.shape)
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, './saved_models/generator.ckpt')

    # fixed_noise = np.random.uniform(low=0., high=1., size=(1, latent_size))

    for i in range(patches_w):
        for j in range(patches_h):
            patch_ij = array_noise[(8*j):(8*j + 8), (8*i):(8*i + 8), :]
            # sample_ij = gen_output.eval(feed_dict={cond_input: patch_ij[None, :], gen_noise: fixed_noise})
            sample_ij = gen_output.eval(feed_dict={cond_input: patch_ij[None, :]})
            reconstructed_image[(8*j):(8*j + 8), (8*i):(8*i + 8), :] = sample_ij

final_im = np.concatenate((array_im, 255*array_noise, 255*reconstructed_image), axis=1)
final_im = np.maximum(final_im, 0)
final_im = np.minimum(final_im, 255)
final_im = Image.fromarray(np.uint8(final_im))
final_im.save('./reconstructed_images/im_sample.jpg', "JPEG")

# Get mean squared error of samples from ground truth
# run w/ noise= 15.94, has 6.31.

import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimage
from resid_conv_cgan import CGAN
import random


def count_in_directory(path, filename):
    files = os.listdir(path)
    return sum(1 for file in files if file.startswith(filename))

# number of patches to average over
num_patches = 10000
sample_directory = os.listdir('./berkeley data/train')
sample_directory = [filename for filename in sample_directory if filename.endswith('.jpg') or filename.endswith('.png')
                    or filename.endswith('.jpeg')]
sample_images = [random.choice(sample_directory) for i in range(num_patches)]

patch_height = 8
patch_width = 8
patch_channels = 3
noise_dev = 255./16.

latent_size = 32  # dimensionality of the noise for the generator
noise_type = tf.random_uniform  # distribution of the noise for the generator
gen_init_shape = (4, 4, 32)
gen_depth = 3
gen_filter_sizes = [(4, 4), (4, 4), (4, 4)]
gen_filter_nums = [32, 24, 16]
gen_strides = [(1, 1), (2, 2), (1, 1)]

cgan = CGAN(patch_height, patch_width, patch_channels, latent_size, noise_type, 0, None, None, None, None,
            gen_init_shape, gen_depth, gen_filter_nums, gen_filter_sizes, gen_strides)

sample_variance = 0

with tf.Session() as sess:
    cond_input, gen_output = cgan.generate(sess, './saved_models/generator')
    for filename in sample_images:
        im = Image.open(os.path.join('./berkeley data/train', filename))
        im_height, im_width = im.size

        window_x = np.random.randint(im_width-patch_width)
        window_y = np.random.randint(im_height-patch_height)
        window = im.crop((window_y, window_x, window_y+patch_height, window_x+patch_width))
        array_patch = mpimage.pil_to_array(window)

        normal_noise = np.random.normal(scale=noise_dev, size=(patch_height, patch_width, patch_channels))
        array_noise = array_patch + normal_noise

        sample = gen_output.eval(feed_dict={cond_input: array_noise[None, :]})
        sample_variance = sample_variance + (1/num_patches)*np.linalg.norm(sample - array_noise)**2

print(np.sqrt(sample_variance))

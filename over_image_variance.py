import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimage
from resid_conv_cgan_2 import CGAN


def count_in_directory(path, filename):
    files = os.listdir(path)
    return sum(1 for file in files if file.startswith(filename))

image_height = 8
image_width = 8
image_channels = 3
noise_type = tf.random_uniform  # distribution of the noise for the generator
gen_depth = 4
gen_filter_sizes = [(4, 4), (4, 4), (4, 4), (4, 4)]
gen_filter_nums = [64, 64, 64, 64]
gen_strides = [(1, 1), (1, 1), (1, 1), (1, 1)]

noise_dev = 255./8.

# name of file in berkeley train directory
file_num = '164074'
image_filename = './berkeley data/train/%s.jpg' % file_num
im = Image.open(image_filename)

# crop for splitting into 8x8 patches
init_width, init_height = im.size
patches_w = init_width//image_width
patches_h = init_height//image_height
im_width = image_width*patches_w
im_height = image_height*patches_h
im = im.crop((0, 0, im_width, im_height))
array_im = mpimage.pil_to_array(im)

np.random.seed(1)
normal_noise = np.random.normal(scale=noise_dev, size=(im_height, im_width, image_channels))
array_noise = array_im + normal_noise
array_noise = (1/255)*array_noise

cgan = CGAN(image_height, image_width, image_channels, noise_type, 0, None, None, None, None,
            gen_depth, gen_filter_nums, gen_filter_sizes, gen_strides)

num_samples = 500

variance_image = np.zeros((im_height, im_width))
max_var = 0
min_var = np.inf
with tf.Session() as sess:
    cond_input, gen_output = cgan.generate(sess, './saved_models/generator')

    for i in range(patches_w):
        for j in range(patches_h):
            patch_ij = array_noise[(8*j):(8*j + 8), (8*i):(8*i + 8), :]
            all_samples = np.zeros((num_samples, image_height, image_width, image_channels))
            for k in range(num_samples):
                sample_k = gen_output.eval(feed_dict={cond_input: patch_ij[None, :]})
                all_samples[k, :] = sample_k
            sample_variance = np.mean(np.var(all_samples, 0), 2)
            variance_image[(8 * j):(8 * j + 8), (8 * i):(8 * i + 8)] = sample_variance
            max_patch = np.amax(sample_variance)
            min_patch = np.amin(sample_variance)
            if (max_patch > max_var):
                max_var = max_patch
            if (min_patch < min_var):
                min_var = min_patch
            print(i*patches_h + j)

variance_image = np.repeat(variance_image[:, :, None], 3, 2)
final_im = np.concatenate((array_im, 255*array_noise, 255*(variance_image-min_var)/(max_var-min_var)), axis=1)
final_im = np.maximum(final_im, 0)
final_im = np.minimum(final_im, 255)
final_im = Image.fromarray(np.uint8(final_im))
final_im.save('./in_progress_images_2/%s_%s.jpg'
              % (file_num, count_in_directory('./in_progress_images_2/', file_num) + 1), "JPEG")

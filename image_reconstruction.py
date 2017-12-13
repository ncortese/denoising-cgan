import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimage
from resid_conv_cgan import CGAN


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

np.random.seed(1)
array_im = mpimage.pil_to_array(im)
normal_noise = np.random.normal(scale=noise_dev, size=(im_height, im_width, image_channels))
array_noise = array_im + normal_noise
array_noise = (1/255)*array_noise

cgan = CGAN(image_height, image_width, image_channels, noise_type, 0, None, None, None, None,
            gen_depth, gen_filter_nums, gen_filter_sizes, gen_strides)

reconstructed_image = np.zeros(array_noise.shape)
with tf.Session() as sess:
    cond_input, gen_output = cgan.generate(sess, './saved_models/generator')

    for i in range(patches_w):
        for j in range(patches_h):
            patch_ij = array_noise[(8*j):(8*j + 8), (8*i):(8*i + 8), :]
            sample_ij = gen_output.eval(feed_dict={cond_input: patch_ij[None, :]})
            reconstructed_image[(8*j):(8*j + 8), (8*i):(8*i + 8), :] = sample_ij

final_im = np.concatenate((array_im, 255*array_noise, 255*reconstructed_image), axis=1)
final_im = np.maximum(final_im, 0)
final_im = np.minimum(final_im, 255)
final_im = Image.fromarray(np.uint8(final_im))
final_im.save('./in_progress_images_2/%s_%s.jpg'
              % (file_num, count_in_directory('./in_progress_images_2/', file_num) + 1), "JPEG")

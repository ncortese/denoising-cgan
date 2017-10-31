import tensorflow as tf
import numpy as np
from PIL import Image
import os
import matplotlib.image as mpimage
from resid_conv_cgan import CGAN


def count_in_directory(path, filename):
    files = os.listdir(path)
    return sum(1 for file in files if file.startswith(filename))

patch_height = 8
patch_width = 8
image_channels = 3
latent_size = 32  # dimensionality of the noise for the generator
noise_type = tf.random_uniform  # distribution of the noise for the generator
gen_init_shape = (4, 4, 32)
gen_depth = 3
gen_filter_sizes = [(4, 4), (4, 4), (4, 4)]
gen_filter_nums = [32, 24, 16]
gen_strides = [(1, 1), (2, 2), (1, 1)]

noise_dev = 255./16.

# name of file in berkeley train directory
file_num = '299091'
image_filename = './berkeley data/train/%s.jpg' % file_num
im = Image.open(image_filename)


im_width, im_height = im.size

np.random.seed(1)
array_im = mpimage.pil_to_array(im)
normal_noise = np.random.normal(scale=noise_dev, size=(im_height, im_width, image_channels))
array_noise = array_im + normal_noise
array_noise = (1/255)*array_noise

cgan = CGAN(patch_height, patch_width, image_channels, latent_size, noise_type, 0, None, None, None, None,
            gen_init_shape, gen_depth, gen_filter_nums, gen_filter_sizes, gen_strides)

reconstructed_image = np.zeros(array_noise.shape)
with tf.Session() as sess:
    cond_input, gen_output = cgan.generate(sess, './saved_models/generator')

    for i in range(im_height-patch_height):
        for j in range(im_width-patch_width):
            patch_ij = array_noise[i:(i+patch_height), j:(j+patch_width), :]
            sample_ij = gen_output.eval(feed_dict={cond_input: patch_ij[None, :]})[0, :, :, :]
            reconstructed_image[i:(i+patch_height), j:(j+patch_width), :] += 1/(patch_height*patch_width)*sample_ij

final_im = np.concatenate((array_im, 255*array_noise, 255*reconstructed_image), axis=1)
final_im = np.maximum(final_im, 0)
final_im = np.minimum(final_im, 255)
final_im = Image.fromarray(np.uint8(final_im))
final_im.save('./reconstructed_images_3/avg_%s_%s.jpg'
              % (file_num, count_in_directory('./reconstructed_images_3/', file_num) + 1), "JPEG")

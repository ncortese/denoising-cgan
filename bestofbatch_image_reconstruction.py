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

batch_size = 10

noise_dev = 255./8.

# name of file in berkeley train directory
file_num = '299091'
image_filename = './berkeley data/train/%s.jpg' % file_num
im = Image.open(image_filename)

# crop for splitting into 8x8 patches
init_width, init_height = im.size
patches_w = init_width//patch_width
patches_h = init_height//patch_height
im_width = patch_width*patches_w
im_height = patch_height*patches_h
im = im.crop((0, 0, im_width, im_height))

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

    for i in range(patches_w):
        for j in range(patches_h):
            patch_ij = array_noise[(patch_height*j):(patch_height*j + 8), (8*i):(8*i + 8), :]
            sample_ij = np.zeros(im)
            min_error = np.inf

            # take sample with least error in batch
            for k in range(batch_size):
                sample_ij_k = gen_output.eval(feed_dict={cond_input: patch_ij[None, :]})
                sample_error = np.linalg.norm(sample_ij_k - array_im[(8*j):(8*j + 8), (8*i):(8*i + 8), :])
                if sample_error < min_error:
                    sample_ij = sample_ij_k
                    min_error = sample_error

            reconstructed_image[(8*j):(8*j + 8), (8*i):(8*i + 8), :] = sample_ij

final_im = np.concatenate((array_im, 255*array_noise, 255*reconstructed_image), axis=1)
final_im = np.maximum(final_im, 0)
final_im = np.minimum(final_im, 255)
final_im = Image.fromarray(np.uint8(final_im))
final_im.save('./reconstructed_images_4/bestofbatch_%s_%s.jpg'
              % (file_num, count_in_directory('./reconstructed_images_3/', file_num) + 1), "JPEG")

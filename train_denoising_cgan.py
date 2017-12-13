from resid_conv_cgan import CGAN
from infinite_denoising_batch_loader import BatchLoader
import tensorflow as tf


# parameters
image_height = 8
image_width = 8
image_channels = 3

noise_dev = 255./8.

batch_size = 64
num_steps = 8000

# latent_size = 32  # dimensionality of the noise for the generator
noise_type = tf.random_uniform  # distribution of the noise for the generator

leaky_relu_alpha = 0.2

disc_depth = 3
disc_filter_sizes = [(4, 4), (3, 3), (3, 3)]
disc_filter_nums = [64, 128, 128]
disc_strides = [(1, 1), (2, 2), (1, 1)]

gen_depth = 4
gen_filter_sizes = [(4, 4), (4, 4), (4, 4), (4, 4)]
gen_filter_nums = [64, 64, 64, 64]
gen_strides = [(1, 1), (1, 1), (1, 1), (1, 1)]

cgan = CGAN(image_height, image_width, image_channels, noise_type, leaky_relu_alpha, disc_depth,
            disc_filter_nums, disc_filter_sizes, disc_strides, gen_depth, gen_filter_nums,
            gen_filter_sizes, gen_strides)

batches = BatchLoader('./berkeley data/train/', (image_height, image_width), image_channels, noise_dev)

# Load from a saved model by cgan.train(batches, batch_size, num_steps, './saved_models/', './generated_samples/', True, n), where n is iteration number of the saved model
cgan.train(batches, batch_size, num_steps, './saved_models/', './generated_samples/', False)

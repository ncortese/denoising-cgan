from PIL import Image
import os
import matplotlib.image as mpimage
import numpy as np
import random


class BatchLoader:
    def __init__(self, patch_directory, noise_directory):
        self.patch_directory = patch_directory
        self.noise_directory = noise_directory
        self.data_n = len(os.listdir(patch_directory))
        self.data_order = np.random.permutation(self.data_n)
        self.index = 0

    def shuffle_data(self):
        self.data_order = random.shuffle(self.data_order)

    def get_patch_i(self, i):
        patch = Image.open(os.path.join(self.patch_directory, str(i) + '.jpg'))
        patch_array = mpimage.pil_to_array(patch)
        return patch_array

    def get_noise_i(self, i):
        noise = Image.open(os.path.join(self.noise_directory, str(i) + '.jpg'))
        noise_array = mpimage.pil_to_array(noise)
        return noise_array

    def get_next_batch(self, batch_size):
        batch_end = min(self.data_n, self.index + batch_size)
        patch_list = [self.get_patch_i(i + 1) for i in range(self.index, batch_end)]
        noise_list = [self.get_noise_i(i + 1) for i in range(self.index, batch_end)]
        if batch_end == self.data_n:
            num_remaining = self.index + batch_size - self.data_n
            self.shuffle_data()
            patch_list = patch_list + [self.get_patch_i(i + 1) for i in range(0, num_remaining)]
            noise_list = noise_list + [self.get_noise_i(i + 1) for i in range(0, num_remaining)]
            self.index = num_remaining

        else:
            self.index = batch_end
        return patch_list, noise_list

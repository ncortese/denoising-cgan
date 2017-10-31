from PIL import Image
import os
import matplotlib.image as mpimage
import numpy as np
import random

# Rather than explicitly making a data set, just gets random patches and adds noise every batch to increase data set
# size without taking up too much space on my computer


class BatchLoader:
    def __init__(self, data_path, patch_size, patch_channels, noise_dev):
        self.data_path = data_path
        self.data_directory = os.listdir(data_path)
        self.data_directory = [filename for filename in self.data_directory if filename.endswith('.jpg')
                               or filename.endswith('.png') or filename.endswith('.jpeg')]
        self.data_n = len(self.data_directory)
        self.patch_size = patch_size
        self.patch_channels = patch_channels
        self.noise_dev = noise_dev

    def get_patch(self):
        i = random.randint(0, self.data_n - 1)
        patch = Image.open((os.path.join(self.data_path, self.data_directory[i])))
        im_width, im_height = patch.size
        window_x = np.random.randint(im_width - self.patch_size[0])
        window_y = np.random.randint(im_height - self.patch_size[1])
        window = patch.crop((window_x, window_y, window_x + self.patch_size[0], window_y + self.patch_size[1]))
        patch_array = mpimage.pil_to_array(window)
        return patch_array

    def get_next_batch(self, batch_size):
        patch_list = [self.get_patch() for i in range(batch_size)]
        noise_list = [patch + np.random.normal(scale=self.noise_dev, size=(self.patch_size + (self.patch_channels,)))
                      for patch in patch_list]
        return patch_list, noise_list

from PIL import Image
import os
import matplotlib.image as mpimage
import numpy as np

# parameter, size of image patches
size = (8, 8)
# parameter, standard deviation of noise
noise_dev = 255./16.
# number of random patches to extract from each image in original data set
num_patches = 50
# minimum variance for a patch to be accepted
min_variance = 200

i = 0
for filename in os.listdir('./berkeley data/train'):
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        im = Image.open(os.path.join('./berkeley data/train', filename))
        im_width, im_height = im.size
        for j in range(num_patches):
            variance = 0
            while variance < min_variance:
                # get a random patch from the image, add noise, and save the original and noisy patches in different
                # directories with the same filename
                window_x = np.random.randint(im_width-size[0])
                window_y = np.random.randint(im_height-size[1])
                window = im.crop((window_x, window_y, window_x+size[0], window_y+size[1]))
                array_patch = mpimage.pil_to_array(window)

                # skip patches with low variance
                variance = np.var(array_patch)

            # could also try making this black and white noise
            normal_noise = np.random.normal(scale=noise_dev, size=(size + (3,)))
            array_noise = array_patch + normal_noise

            patch = Image.fromarray(np.uint8(array_patch))
            noise = Image.fromarray(np.uint8(array_noise))
            patch.save(os.path.join('./berkeley_patches', "%s.jpg" % str(num_patches*i + j)), "JPEG")
            noise.save(os.path.join('./berkeley_noise', "%s.jpg" % str(num_patches*i + j)), "JPEG")
        i += 1
        print(i)

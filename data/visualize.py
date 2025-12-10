# data/visualize.py

import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from config import TRAIN_DIR

def show_samples(classes=['A', 'B', 'C'], n_per_class=2):
    plt.figure(figsize=(15, 5))
    
    for i, folder in enumerate(classes):
        folder_path = os.path.join(TRAIN_DIR, folder)
        images = random.sample(os.listdir(folder_path), n_per_class)

        for j, img_name in enumerate(images):
            img = mpimg.imread(os.path.join(folder_path, img_name))
            plt.subplot(1, len(classes) * n_per_class, i * n_per_class + j + 1)
            plt.imshow(img)
            plt.title(folder)
            plt.axis("off")

    plt.show()

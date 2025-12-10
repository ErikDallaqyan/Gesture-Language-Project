# data/dataloader.py

import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from config import TRAIN_DIR, TEST_DIR, IMAGE_SIZE, BATCH_SIZE

def load_data():
    train_gen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    training_set = train_gen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    test_set = test_gen.flow_from_directory(
        TEST_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical'
    )

    return training_set, test_set

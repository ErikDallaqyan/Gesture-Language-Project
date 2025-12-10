# models/cnn_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from config import IMAGE_SIZE, LEARNING_RATE, NUM_CLASSES

def build_model():
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(64, 3, 3, activation='relu', padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(128, 3, 3, activation='relu', padding="same"))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='softmax'))

    model.compile(
        optimizer=optimizers.SGD(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

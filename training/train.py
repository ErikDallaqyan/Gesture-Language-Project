# training/train.py

from config import EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS
from models.cnn_model import build_model
from data.dataloader import load_data

def train():
    training_set, test_set = load_data()
    model = build_model()

    history = model.fit(
        training_set,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=test_set,
        validation_steps=VALIDATION_STEPS
    )

    model.save("Trained_model.h5")
    return history, training_set


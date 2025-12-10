import matplotlib.pyplot as plt

def plot(history):
    # Accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"])
    plt.grid()
    plt.show()

    # Loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "test"])
    plt.grid()
    plt.show()

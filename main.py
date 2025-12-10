from utils.unzip import unzip_dataset
from training.train import train
from evaluation.plot_history import plot
from evaluation.save_labels import save_labels
from data.visualize import show_samples
from config import DATASET_ZIP

def main():

    unzip_dataset(DATASET_ZIP)
    show_samples()
    history, training_set = train()

    plot(history.history)
    save_labels(training_set)

if __name__ == "__main__":
    main()

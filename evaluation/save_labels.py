# save labels
import json

def save_labels(training_set):
    labels = training_set.class_indices
    inv_labels = {v: k for k, v in labels.items()}

    with open("labels.json", "w") as f:
        json.dump(inv_labels, f)

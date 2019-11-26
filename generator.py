import cv2
import os
import numpy as np

label_map = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]), #[0, 0, 0, 0],
    1: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]), #[0, 0, 0, 1],
    10: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]), #[0, 0, 1, 0],
    100: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]), #[0, 1, 0, 0],
    101: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]), #[0, 1, 0, 1],
    110: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]), #[0, 1, 1, 0],
    1000: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]), #[1, 0, 0, 0],
    1001: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]), #[1, 0, 0, 1],
    1010: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]), #[1, 0, 1, 0]
}

def read_labels(file):
    with open(file) as f:
        lines = f.read().splitlines()

    labels = {}
    for i, line in enumerate(lines):
        labels[i] = line
    return labels

def generate_inputs(folder, labels_file, sequence_size, IMG_HEIGHT, IMG_WIDTH):
    """Create input generators for given parameters"""
    # TODO Needs to handle resting/non-resting and every Nth frame
    # Get labels
    labels = read_labels(labels_file)
    # Enumerate files in directory
    available_ids = range(len(os.listdir(folder)))
    while True:
        # From first item to last possible (taking sequence length into consideration)
        for i in available_ids[:-sequence_size]:
            sequence = []
            # Read in image and add it to list for the current sequence
            for j in range(sequence_size):
                path = '{}/{}.jpg'.format(folder, i+j)
                image = cv2.imread(path)
                res = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
                sequence.append(res)
            # Get label and return sequence with corresponding label
            val = int(labels[i+j])
            label = label_map[val]
            yield (np.array([sequence]), np.array([label]))

def chain_generators(generators):
    """Concatenate multiple generators together. From: https://stackoverflow.com/a/47592164"""
    for gen in generators:
        yield from gen

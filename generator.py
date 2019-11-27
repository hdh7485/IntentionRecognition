import cv2
import os
import numpy as np

label_map = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0, 1]),  # [0, 0, 0, 0],
    1: np.array([0, 0, 0, 0, 0, 0, 0, 1, 0]),  # [0, 0, 0, 1],
    10: np.array([0, 0, 0, 0, 0, 0, 1, 0, 0]),  # [0, 0, 1, 0],
    100: np.array([0, 0, 0, 0, 0, 1, 0, 0, 0]),  # [0, 1, 0, 0],
    101: np.array([0, 0, 0, 0, 1, 0, 0, 0, 0]),  # [0, 1, 0, 1],
    110: np.array([0, 0, 0, 1, 0, 0, 0, 0, 0]),  # [0, 1, 1, 0],
    1000: np.array([0, 0, 1, 0, 0, 0, 0, 0, 0]),  # [1, 0, 0, 0],
    1001: np.array([0, 1, 0, 0, 0, 0, 0, 0, 0]),  # [1, 0, 0, 1],
    1010: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0]),  # [1, 0, 1, 0]
}


def read_labels(file):
    with open(file) as f:
        lines = f.read().splitlines()

    labels = {}
    for i, line in enumerate(lines):
        labels[i] = line
    return labels


def generate_inputs(folder, labels_file, sequence_size, IMG_HEIGHT, IMG_WIDTH, non_resting=False, skip_frames=1):
    """Create input generators for given parameters"""
    # Get labels
    labels = read_labels(labels_file)
    # Enumerate files in directory
    available_ids = range(len(os.listdir(folder)))
    while True:
        # From first item to last possible (taking sequence length into consideration)
        for i in available_ids[:- (sequence_size-1) * skip_frames]:
            sequence = []
            # Read in image and add it to list for the current sequence
            for j in range(sequence_size):
                path = '{}/{}.jpg'.format(folder, i + x * j)
                image = cv2.imread(path)
                res = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
                sequence.append(res)
            val = int(labels[i + x * j])
            # Get label and return sequence with corresponding label
            if non_resting:
                if val != 0:
                    label = dic[val]
                    yield (np.array([sequence]), np.array([label]))
            else:
                label = label_map[val]
                yield (np.array([sequence]), np.array([label]))


def chain_generators(generators):
    """Concatenate multiple generators together. From: https://stackoverflow.com/a/47592164"""
    for gen in generators:
        yield from gen

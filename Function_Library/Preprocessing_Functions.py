
import numpy as np


def normalize(x: np.ndarray) -> np.ndarray:
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def generate_onehot(label, possible_labels):
    outlist = []
    for i in range(possible_labels):
        if label == i:
            outlist.append(1)
        else:
            outlist.append(0)

    outlist2 = np.array(outlist)
    return outlist2

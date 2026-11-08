import numpy as np

def normalize(img):
    mean = np.mean(img)
    std = np.std(img) + 1e-06
    return (img - mean) / std

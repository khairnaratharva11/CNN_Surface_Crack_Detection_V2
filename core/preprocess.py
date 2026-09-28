import cv2
import numpy as np

def preprocess_patch(patch, target_size):
    """
    Preprocess a single patch identically to training:
    - resize to target_size (e.g. 128x128)
    - convert to float32
    - rescale by 1/255.0
    """
    patch = cv2.resize(patch, target_size)
    patch = patch.astype(np.float32) / 255.0
    return patch

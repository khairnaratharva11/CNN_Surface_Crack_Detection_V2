import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

def measure_crack(patch_mask):
    """
    Computes measurements for a crack given its binary mask patch.
    Returns skeleton, length_px, mean_width_px, max_width_px, area_px
    """
    skel = skeletonize(patch_mask > 0)
    length_px = np.sum(skel)
    
    dist_transform = distance_transform_edt(patch_mask)
    widths = dist_transform[skel] * 2
    
    if len(widths) > 0:
        mean_width_px = float(np.mean(widths))
        max_width_px = float(np.max(widths))
    else:
        mean_width_px = 0.0
        max_width_px = 0.0
        
    area_px = np.count_nonzero(patch_mask)

    
    return skel, length_px, mean_width_px, max_width_px, area_px

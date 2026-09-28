import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

class SeverityAssessor:
    def __init__(self, config):
        self.config = config['severity']
        self.calib_px_per_mm = config['camera']['calibration_pixels_per_mm']
        
    def assess(self, detection, frame_shape):
        """
        Computes severity metrics for a single crack detection.
        Modifies the detection dict in-place by adding a 'severity' dictionary.
        """
        skel = detection['skeleton']
        patch_mask = detection['patch_mask']
        
        # 1. Length: Count skeleton pixels
        length_px = np.sum(skel)
        
        # 2. Width: Distance transform of the mask
        # Distance transform gives distance to the nearest zero pixel (background).
        # We multiply by 2 to get full width.
        dist_transform = distance_transform_edt(patch_mask)
        widths = dist_transform[skel] * 2
        
        if len(widths) > 0:
            mean_width_px = float(np.mean(widths))
            max_width_px = float(np.max(widths))
        else:
            mean_width_px = 0.0
            max_width_px = 0.0
            
        # 3. Area ratio
        area_px = np.sum(patch_mask)
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = float(area_px / frame_area)
        
        # Calculate real-world units if calibrated
        if self.calib_px_per_mm:
            length = length_px / self.calib_px_per_mm
            mean_width = mean_width_px / self.calib_px_per_mm
            max_width = max_width_px / self.calib_px_per_mm
            unit = 'mm'
        else:
            length = float(length_px)
            mean_width = mean_width_px
            max_width = max_width_px
            unit = 'px'
            
        # Determine Severity Class based on config
        # Defaulting to length for simplicity as per rules
        l_low = self.config['length_low']
        l_mod = self.config['length_moderate']
        l_high = self.config['length_high']
        
        if length_px >= l_high:
            severity_class = "Critical"
        elif length_px >= l_mod:
            severity_class = "High"
        elif length_px >= l_low:
            severity_class = "Moderate"
        else:
            severity_class = "Low"
            
        action = self.config['actions'].get(severity_class, "Unknown")
        
        detection['severity_stats'] = {
            'length': length,
            'mean_width': mean_width,
            'max_width': max_width,
            'area_ratio': area_ratio,
            'unit': unit,
            'severity_class': severity_class,
            'action': action
        }
        
        return detection['severity_stats']

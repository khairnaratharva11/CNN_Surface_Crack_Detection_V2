import pytest
import numpy as np
from src.inference.severity import SeverityAssessor

def test_severity_assessment():
    config = {
        'severity': {
            'length_low': 50,
            'length_moderate': 100,
            'length_high': 200,
            'actions': {
                'Low': 'Monitor',
                'Moderate': 'Schedule sealing',
                'High': 'Repair soon',
                'Critical': 'Immediate inspection'
            }
        },
        'camera': {
            'calibration_pixels_per_mm': None
        }
    }
    
    assessor = SeverityAssessor(config)
    
    # Mock a detection
    skel = np.zeros((100, 100), dtype=bool)
    skel[10:90, 50] = True # Length = 80
    
    patch_mask = np.zeros((100, 100), dtype=np.uint8)
    patch_mask[10:90, 48:53] = 255 # Width = 5 (2 pixels left, 2 pixels right + center)
    
    detection = {
        'skeleton': skel,
        'patch_mask': patch_mask
    }
    
    frame_shape = (1080, 1920, 3)
    
    stats = assessor.assess(detection, frame_shape)
    
    assert stats['length'] == 80.0
    assert stats['unit'] == 'px'
    # Width is derived from distance transform. Center line has dist ~2.5, width ~5
    assert stats['mean_width'] > 4.0 and stats['mean_width'] < 6.0
    assert stats['severity_class'] == 'Moderate' # 80 is between 50 and 100
    assert stats['action'] == 'Schedule sealing'

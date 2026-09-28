import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
from core.measure import measure_crack
from core.tracker import compute_iou, CrackTracker

def test_measurement_synthetic():
    # Create a synthetic crack: a straight line of width 3 and length 10
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 8:11] = 255 # Length 10, Width 3
    
    skel, length_px, mean_width_px, max_width_px, area_px = measure_crack(mask)
    
    assert area_px == 30
    assert length_px >= 8 and length_px <= 10 # Skeleton length can vary slightly
    assert max_width_px >= 2.0 and max_width_px <= 4.0

def test_iou():
    boxA = (0, 0, 10, 10)
    boxB = (0, 0, 10, 10)
    assert compute_iou(boxA, boxB) == pytest.approx(1.0, rel=1e-4)
    
    boxC = (10, 10, 10, 10)
    assert compute_iou(boxA, boxC) == 0.0

def test_tracker_shift():
    tracker = CrackTracker(min_hits=1)
    
    # Frame 1
    det1 = [{'bbox': (100, 100, 20, 20), 'data': 'crack1'}]
    tracks1 = tracker.update(det1, global_shift=(0, 0))
    assert len(tracks1) == 1
    crack_id = tracks1[0]['crack_id']
    
    # Frame 2: camera panned, so the crack moved by dx=10, dy=0
    # The detector will find it at (90, 100)
    det2 = [{'bbox': (90, 100, 20, 20), 'data': 'crack1_shifted'}]
    
    # But we pass the global shift we estimated: dx=10, dy=0 (background moved left, so shift to apply to old tracks is dx=-10?
    # Wait, if camera pans right, background moves left. Crack moves left. 
    # If crack is now at 90, old was 100.
    # The tracker adds shift to old boxes: 100 + (-10) = 90.
    tracks2 = tracker.update(det2, global_shift=(-10, 0))
    
    assert len(tracks2) == 1
    assert tracks2[0]['crack_id'] == crack_id # ID should be preserved

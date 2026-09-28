import pytest
import numpy as np
import cv2
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.comparison import compute_session_summary, compute_priority_score
from core.color_analysis import extract_dominant_colors
from core.deepseek_material import analyze_material

def test_priority_score():
    # Mock summary
    summary = {
        "total_cracks": 10,
        "counts_by_severity": {"Critical": 1, "High": 2, "Moderate": 5, "Low": 2},
        "max_severity_score": 90.0,
        "mean_severity_score": 45.0,
        "density": 5.0, # cracks per min
        "unreviewed_ratio": 1.0
    }
    
    score = compute_priority_score(summary)
    assert score > 0
    assert score <= 100.0

def test_color_extraction(tmp_path):
    # Create a synthetic solid color image (e.g., Brick Red [178, 34, 34])
    # In BGR for cv2.imwrite:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:] = (34, 34, 178) # BGR
    
    img_path = str(tmp_path / "test_color.jpg")
    cv2.imwrite(img_path, img)
    
    res = extract_dominant_colors(img_path)
    assert res is not None
    assert res['nearest_color_name'] == "Brick Red"
    
def test_deepseek_parsing(monkeypatch, tmp_path):
    # Mock requests.post
    class MockResponse:
        def __init__(self, json_data, status_code=200):
            self.json_data = json_data
            self.status_code = status_code
        def raise_for_status(self):
            pass
        def json(self):
            return self.json_data
            
    def mock_post(*args, **kwargs):
        return MockResponse({
            "choices": [{
                "message": {
                    "content": "```json\n{\"material\": \"concrete\", \"confidence\": 0.9, \"notes\": \"Needs epoxy\", \"repair_materials\": [\"Epoxy\"], \"repair_tools\": [\"Trowel\"]}\n```"
                }
            }]
        })
        
    monkeypatch.setattr("requests.post", mock_post)
    
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img_path = str(tmp_path / "test.jpg")
    cv2.imwrite(img_path, img)
    
    res = analyze_material(img_path, "fake_key")
    assert res['material_label'] == "concrete"
    assert res['material_confidence'] == 0.9
    assert res['repair_materials'] == ["Epoxy"]

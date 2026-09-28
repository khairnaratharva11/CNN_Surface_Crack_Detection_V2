import streamlit as st
import cv2
import numpy as np
from core.config import config
from core.model_loader import load_crack_model
from core.localizer import TiledClassifierLocalizer
from core.measure import measure_crack
from core.severity import SeverityAssessor
import time

st.set_page_config(page_title="Image Inspect", page_icon="🖼️", layout="wide")
st.title("Single Image Inspection")

with st.spinner("Loading Model..."):
    model = load_crack_model(config['model']['arch'], config['model']['image_size'], config['paths']['models_dir'])
    localizer = TiledClassifierLocalizer(model, config)
    assessor = SeverityAssessor(config)

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Uploaded Image", use_column_width=True)
    
    if st.button("Detect Cracks"):
        with st.spinner("Processing..."):
            start = time.time()
            detections = localizer.detect(frame)
            
            for det in detections:
                skel, length_px, mean_width_px, max_width_px, area_px = measure_crack(det['patch_mask'])
                sev_stats = assessor.assess(length_px, mean_width_px, max_width_px, area_px, det['confidence'], frame.shape)
                det['severity_stats'] = sev_stats
                
            elapsed = time.time() - start
            
        st.success(f"Found {len(detections)} cracks in {elapsed:.2f}s")
        
        display_frame = frame.copy()
        colors = {
            "Low": (0, 255, 0),
            "Moderate": (0, 255, 255),
            "High": (255, 165, 0),
            "Critical": (255, 0, 0)
        }
        
        for det in detections:
            sev = det['severity_stats']['severity_class']
            color = colors.get(sev, (255, 255, 255))
            x, y, w, h = det['bbox']
            cv2.rectangle(display_frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
            
        st.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), caption="Detections", use_column_width=True)

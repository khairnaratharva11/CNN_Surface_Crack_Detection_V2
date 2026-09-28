import streamlit as st
import cv2
import tempfile
import os
import time
from core.config import config
from core.model_loader import load_crack_model
from core.localizer import TiledClassifierLocalizer
from core.video_pipeline import VideoPipeline

st.set_page_config(page_title="Video Analysis", page_icon="🎞️", layout="wide")
st.title("Offline Video Analysis")

with st.spinner("Loading Model..."):
    model = load_crack_model(config['model']['arch'], config['model']['image_size'], config['paths']['models_dir'])
    localizer = TiledClassifierLocalizer(model, config)

st.sidebar.header("Settings")
site_tag = st.sidebar.text_input("Site Tag", value="Offline Video")
operator = st.sidebar.text_input("Operator", value="Inspector 1")

uploaded_file = st.file_uploader("Upload a Video File", type=["mp4", "avi", "mkv"])

if uploaded_file is not None:
    # Save temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()
    
    st.video(tfile.name)
    
    if st.button("Start Analysis"):
        st.write("Processing...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        pipeline = VideoPipeline(localizer, config)
        pipeline.start_session(source_type="VideoFile", source_path=uploaded_file.name, site_tag=site_tag, operator=operator)
        
        cap = cv2.VideoCapture(tfile.name)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        start_time = time.time()
        
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            pipeline.process_frame(frame)
            frame_idx += 1
            
            if frame_idx % 10 == 0:
                progress = min(1.0, frame_idx / total_frames)
                progress_bar.progress(progress)
                elapsed = time.time() - start_time
                fps = frame_idx / elapsed
                status_text.text(f"Processed {frame_idx}/{total_frames} frames (FPS: {fps:.1f})")
                
        cap.release()
        pipeline.end_session()
        
        progress_bar.progress(1.0)
        status_text.text("Analysis Complete!")
        os.unlink(tfile.name)
        st.success(f"Results saved to Database. Session ID: {pipeline.session_id}")

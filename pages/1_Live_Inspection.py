import streamlit as st
import cv2
import threading
import queue
import time
from core.config import config
from core.model_loader import load_crack_model
from core.localizer import TiledClassifierLocalizer
from core.video_pipeline import VideoPipeline

st.set_page_config(page_title="Live Inspection", page_icon="🎥", layout="wide")
st.title("Live Video Inspection")

# Load Model
with st.spinner("Loading Model..."):
    model = load_crack_model(config['model']['arch'], config['model']['image_size'], config['paths']['models_dir'])
    localizer = TiledClassifierLocalizer(model, config)

# Sidebar
st.sidebar.header("Settings")
source_idx = st.sidebar.selectbox("Camera Source", [0, 1, 2], index=config['camera']['default_source'])
site_tag = st.sidebar.text_input("Site Tag", value="Site A - Pier 1")
operator = st.sidebar.text_input("Operator", value="Inspector 1")

# Overlays
overlay_boxes = st.sidebar.checkbox("Show Bounding Boxes", value=True)
overlay_mask = st.sidebar.checkbox("Show Mask Overlay", value=False)
overlay_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)

# UI Elements
col1, col2 = st.columns([3, 1])

with col1:
    stframe = st.empty()
    
with col2:
    st.subheader("Session Stats")
    fps_metric = st.empty()
    cracks_metric = st.empty()
    critical_metric = st.empty()
    event_feed = st.empty()

# State
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

def start_stream():
    st.session_state.is_running = True
    
def stop_stream():
    st.session_state.is_running = False

col_start, col_stop = st.sidebar.columns(2)
col_start.button("Start", on_click=start_stream, use_container_width=True)
col_stop.button("Stop", on_click=stop_stream, use_container_width=True)

if st.session_state.is_running:
    # Initialize pipeline
    pipeline = VideoPipeline(localizer, config)
    pipeline.start_session(source_type="Webcam", source_path=str(source_idx), site_tag=site_tag, operator=operator)
    
    cap = cv2.VideoCapture(source_idx)
    
    # We use a simple loop in the main thread for Streamlit, as Streamlit state doesn't update well from bg threads
    # However, to prevent UI lagging behind camera, we can clear the buffer
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    total_cracks = 0
    critical_cracks = 0
    event_logs = []
    
    prev_time = time.time()
    
    try:
        while st.session_state.is_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera.")
                break
                
            # Process
            tracks, _ = pipeline.process_frame(frame)
            
            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-5)
            prev_time = curr_time
            
            # Draw
            display_frame = frame.copy()
            total_cracks = len(tracks)
            critical_cracks = sum(1 for t in tracks if t['severity_stats']['severity_class'] == 'Critical')
            
            colors = {
                "Low": (0, 255, 0),
                "Moderate": (0, 255, 255),
                "High": (255, 165, 0),
                "Critical": (255, 0, 0) # Red in RGB
            }
            
            for t in tracks:
                sev = t['severity_stats']['severity_class']
                color = colors.get(sev, (255, 255, 255))
                
                if overlay_boxes:
                    x, y, w, h = t['bbox']
                    # OpenCV is BGR usually, but we will convert to RGB for Streamlit
                    cv2.rectangle(display_frame, (int(x), int(y)), (int(x+w), int(y+h)), color, 2)
                    label = f"ID:{t['crack_id']} {sev} {t['severity_stats']['max_width']:.1f}{t['severity_stats']['unit']}"
                    cv2.putText(display_frame, label, (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
            # Convert to RGB for Streamlit
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            stframe.image(display_frame, channels="RGB", use_column_width=True)
            
            # Update metrics
            fps_metric.metric("FPS", f"{fps:.1f}")
            cracks_metric.metric("Active Cracks", total_cracks)
            critical_metric.metric("Critical", critical_cracks)
            
            # This allows Streamlit to process button clicks
            time.sleep(0.01)
            
    finally:
        cap.release()
        pipeline.end_session()
        st.success("Session Ended and Saved.")

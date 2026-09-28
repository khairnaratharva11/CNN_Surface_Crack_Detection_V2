import streamlit as st

st.set_page_config(
    page_title="Crack Detection App",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("CNN Surface Crack Detection V2")

st.markdown("""
Welcome to the Industrial Surface Crack Detection system.

### Features
- **Live Inspection**: Process webcam or RTSP streams in real-time.
- **Video Analysis**: Upload video files for offline processing.
- **Crack Registry**: View, filter, and manage detected cracks across all sessions.
- **Reports**: Generate PDF/CSV inspection reports.
- **Image Inspect**: Single image and folder batch processing.
- **Model Insights**: View model metrics and metadata.

👈 Select a tool from the sidebar to get started.

> **DISCLAIMER:** This is a model assistance tool and is NOT a substitute for certified structural inspection.
""")

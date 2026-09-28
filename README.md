# CNN Surface Crack Detection V2 (Streamlit App)

A robust, real-time desktop application for industrial surface crack detection, now upgraded to a modern, responsive **Streamlit** architecture.

## Features
- **Real-Time Live Inspection**: Process webcam or RTSP streams directly in your browser without UI freezing.
- **Offline Video Analysis**: Upload and process video files with progress tracking.
- **Severity Assessment & Tracking**: Computes crack length/width, assigns severity (Low to Critical), and tracks cracks across frames to prevent duplicate logging.
- **Crack Registry & SQLite DB**: Every detected crack is permanently logged to `db/cracks.db` for later review, filtering, and status updates (e.g., "repaired").
- **Reporting**: Export session results to CSV and professional PDF reports.
- **Compare & Prioritize**: Ranks multiple inspection sessions automatically by severity score and crack density.
- **DeepSeek Surface Analysis**: Uses the DeepSeek Vision API to detect material types and required repair tools for individual cracks.
- **CPU-Optimized Inference:** Uses a MobileNetV2 patch classifier combined with tiled sliding-window localization.

## Architecture
- `core/`: Pure Python modules for localization, measurement, tracking, and video processing.
- `ui/pages/`: Streamlit web interface pages.
- `db/`: SQLite schemas and access patterns.
- `tests/`: Automated test suite for measurements and tracking logic.

### How it Works
1. **Tiled Inference**: The full frame is split into overlapping patches. Each patch is classified by the MobileNetV2 model to generate a probability heatmap.
2. **Measurement**: The heatmap is thresholded into a binary mask. The mask is skeletonized to measure length, and a distance transform measures the true width.
3. **Severity**: A weighted score is computed from width, length, and confidence.
4. **Tracking**: An IoU and centroid-based tracker (with optical-flow shift compensation) ensures a crack is only logged once per physical defect.

## Setup
1. Install Python 3.10+.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure your trained `.keras` model is placed in the `models/` directory.
4. Review thresholds in `config.yaml` or via the **Settings** page in the app.

## Usage
Launch the Streamlit dashboard:
```bash
./run.sh
# or on Windows:
run.bat
```
Then navigate to `http://localhost:8501` in your browser.

## Calibration Guide
By default, the app measures cracks in **pixels** and reports an "Uncalibrated" relative severity.
To measure in millimeters:
1. Navigate to the **Settings** page.
2. Enter the known `pixels_per_mm` for your typical camera distance.
3. Save settings. Measurements will now be reported in `mm` and used for strict severity grading.

## Known Limitations
- **Camera Motion**: The tracker handles slow panning via phase correlation, but fast, blurry movement may cause ID switching and duplicate logging.
- **Uncalibrated Accuracy**: Pixel measurements do not translate to physical severity if the camera distance varies significantly without re-calibration.
- **False Positives**: Extreme surface textures may trigger detections. Use the Grad-CAM/Heatmap overlays to debug, and mark false positives in the Crack Registry.

import sys
import os
import cv2
import yaml
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSlider, QTableWidget, 
    QTableWidgetItem, QHeaderView, QGroupBox, QSplitter
)
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QImage, QPixmap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from src.app.ui_components import VideoStreamThread, InferenceThread
from src.inference.localizer import Localizer
from src.inference.severity import SeverityAssessor
from src.models.architectures import get_model

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

class CrackDetectionApp(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("CNN Surface Crack Detection V2")
        self.setGeometry(100, 100, 1200, 800)
        
        self.video_thread = None
        self.inference_thread = None
        
        # Load Model
        self.init_model()
        
        # UI Setup
        self.init_ui()
        
    def init_model(self):
        arch_name = self.config['model']['arch']
        models_dir = self.config['paths']['models_dir']
        model_path = os.path.join(models_dir, f'{arch_name}_finetuned_best.keras')
        
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f'{arch_name}_best.keras')
            
        print(f"Loading model: {model_path}")
        self.model = get_model(arch_name, tuple(self.config['model']['image_size']) + (3,))
        if os.path.exists(model_path):
            self.model.load_weights(model_path)
            
        self.localizer = Localizer(self.model, self.config)
        self.assessor = SeverityAssessor(self.config)
        
    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Left Panel (Video)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        self.video_label = QLabel("Video Feed")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setMinimumSize(640, 480)
        left_layout.addWidget(self.video_label)
        
        controls_layout = QHBoxLayout()
        
        self.source_combo = QComboBox()
        self.source_combo.addItems(["Webcam (0)", "Webcam (1)", "Choose Video File..."])
        controls_layout.addWidget(self.source_combo)
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.toggle_detection)
        controls_layout.addWidget(self.start_btn)
        
        self.overlay_combo = QComboBox()
        self.overlay_combo.addItems(["Bounding Boxes", "Heatmap", "Segmentation Mask"])
        controls_layout.addWidget(QLabel("Overlay:"))
        controls_layout.addWidget(self.overlay_combo)
        
        left_layout.addLayout(controls_layout)
        
        # Right Panel (Stats & List)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.stats_label = QLabel("Total Cracks: 0 | Critical: 0")
        right_layout.addWidget(self.stats_label)
        
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["ID", "Severity", "Length", "Confidence", "Action"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        right_layout.addWidget(self.table)
        
        # Add to main
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([800, 400])
        main_layout.addWidget(splitter)
        
    @Slot()
    def toggle_detection(self):
        if self.video_thread is None or not self.video_thread.running:
            # Start
            idx = self.source_combo.currentIndex()
            if idx == 0:
                source = 0
            elif idx == 1:
                source = 1
            else:
                from PySide6.QtWidgets import QFileDialog
                file_name, _ = QFileDialog.getOpenFileName(self, "Open Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
                if not file_name:
                    return # User cancelled
                source = file_name
            
            self.video_thread = VideoStreamThread(source)
            self.video_thread.frame_ready.connect(self.on_frame_ready)
            
            self.inference_thread = InferenceThread(self.localizer, self.assessor)
            self.inference_thread.result_ready.connect(self.on_result_ready)
            
            self.video_thread.start()
            self.inference_thread.start()
            self.start_btn.setText("Stop Detection")
        else:
            # Stop
            self.video_thread.stop()
            self.inference_thread.stop()
            self.start_btn.setText("Start Detection")
            self.video_label.clear()
            
    @Slot(np.ndarray)
    def on_frame_ready(self, frame):
        if self.inference_thread and self.inference_thread.running:
            self.inference_thread.add_frame(frame)
            
    @Slot(np.ndarray, list, np.ndarray, np.ndarray)
    def on_result_ready(self, frame, detections, heatmap, mask):
        # Determine active overlay
        overlay = self.overlay_combo.currentText()
        out_frame = frame.copy()
        
        if overlay == "Heatmap":
            hm_colored = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_JET)
            out_frame = cv2.addWeighted(out_frame, 0.6, hm_colored, 0.4, 0)
        elif overlay == "Segmentation Mask":
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mask_colored[:, :, 0] = 0 # yellow/red tint
            out_frame = cv2.addWeighted(out_frame, 0.7, mask_colored, 0.3, 0)
            
        colors = {
            "Low": (0, 255, 0),
            "Moderate": (0, 255, 255),
            "High": (0, 165, 255),
            "Critical": (0, 0, 255)
        }
        
        self.table.setRowCount(len(detections))
        critical_count = 0
        
        for i, det in enumerate(detections):
            stats = det['severity_stats']
            sev = stats['severity_class']
            color = colors.get(sev, (255, 255, 255))
            
            if sev == "Critical":
                critical_count += 1
            
            if overlay == "Bounding Boxes":
                x, y, w, h = det['bbox']
                cv2.rectangle(out_frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(out_frame, f"ID:{det['id']} {sev}", (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
                # Draw skeleton
                skel_pts = np.argwhere(det['skeleton'] > 0)
                for pt in skel_pts:
                    # pt is (y, x) inside the patch
                    cv2.circle(out_frame, (x + pt[1], y + pt[0]), 1, (255, 0, 0), -1)
            
            # Update Table
            self.table.setItem(i, 0, QTableWidgetItem(str(det['id'])))
            self.table.setItem(i, 1, QTableWidgetItem(sev))
            self.table.setItem(i, 2, QTableWidgetItem(f"{stats['length']:.1f} {stats['unit']}"))
            self.table.setItem(i, 3, QTableWidgetItem(f"{det['confidence']:.2f}"))
            self.table.setItem(i, 4, QTableWidgetItem(stats['action']))
            
        self.stats_label.setText(f"Total Cracks: {len(detections)} | Critical: {critical_count}")
        
        # Convert to QPixmap
        h, w, ch = out_frame.shape
        bytes_per_line = ch * w
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qimg).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def closeEvent(self, event):
        if self.video_thread:
            self.video_thread.stop()
        if self.inference_thread:
            self.inference_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.path)
    config = load_config()
    window = CrackDetectionApp(config)
    window.show()
    sys.exit(app.exec())

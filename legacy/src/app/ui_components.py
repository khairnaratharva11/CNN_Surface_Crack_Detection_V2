import sys
import os
import cv2
import numpy as np
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QComboBox, QSlider, QTableWidget, 
    QTableWidgetItem, QHeaderView, QGroupBox, QSplitter
)
from PySide6.QtCore import Qt, QTimer, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QFont

class VideoStreamThread(QThread):
    frame_ready = Signal(np.ndarray)
    
    def __init__(self, source=0):
        super().__init__()
        self.source = source
        self.running = False
        self.cap = None
        
    def run(self):
        self.cap = cv2.VideoCapture(self.source)
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.frame_ready.emit(frame)
            else:
                self.msleep(100) # Wait a bit if file ended or cam lost
                
    def stop(self):
        self.running = False
        self.wait()
        if self.cap:
            self.cap.release()

class InferenceThread(QThread):
    result_ready = Signal(np.ndarray, list, np.ndarray, np.ndarray) # frame_out, detections, heatmap, mask
    
    def __init__(self, localizer, assessor):
        super().__init__()
        self.localizer = localizer
        self.assessor = assessor
        self.frame_queue = []
        self.running = False
        
    def add_frame(self, frame):
        if len(self.frame_queue) < 2:
            self.frame_queue.append(frame)
            
    def run(self):
        self.running = True
        while self.running:
            if self.frame_queue:
                frame = self.frame_queue.pop(0)
                detections, heatmap, mask = self.localizer.process_frame(frame)
                
                for det in detections:
                    self.assessor.assess(det, frame.shape)
                    
                self.result_ready.emit(frame, detections, heatmap, mask)
            else:
                self.msleep(10)
                
    def stop(self):
        self.running = False
        self.wait()

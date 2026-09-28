import cv2
import time
import numpy as np
import os
from uuid import uuid4
from core.measure import measure_crack
from core.severity import SeverityAssessor
from core.tracker import CrackTracker
from db.database import Database

class VideoPipeline:
    def __init__(self, localizer, config, session_id=None):
        self.localizer = localizer
        self.config = config
        self.session_id = session_id or str(uuid4())
        
        self.assessor = SeverityAssessor(config)
        
        # Tracking config
        t_cfg = config.get('tracking', {})
        self.tracker = CrackTracker(
            max_age=t_cfg.get('max_age', 15),
            iou_threshold=t_cfg.get('iou_threshold', 0.3),
            min_hits=t_cfg.get('min_hits', 3)
        )
        
        self.process_every_n = t_cfg.get('process_every_nth_frame', 2)
        
        # Database
        self.db = Database(config['database']['path'])
        
        self.frame_count = 0
        self.start_time = time.time()
        
        # For camera shift compensation
        self.prev_gray = None
        
        # State
        self.is_running = True
        
    def start_session(self, source_type, source_path, site_tag, operator):
        self.db.start_session(
            session_id=self.session_id,
            source_type=source_type,
            source_path=source_path,
            site_tag=site_tag,
            operator=operator,
            mm_per_pixel=self.config['camera'].get('calibration_pixels_per_mm'),
            model_version=self.config['model']['arch'],
            settings=self.config
        )
        
    def process_frame(self, frame, timestamp=None):
        if timestamp is None:
            timestamp = time.time() - self.start_time
            
        self.frame_count += 1
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Camera shift compensation (very simple optical flow / phase correlate)
        dx, dy = 0, 0
        if self.prev_gray is not None:
            # Using phase correlation for global translation
            # For performance, we could do it on a downscaled image
            small_prev = cv2.resize(self.prev_gray, (0,0), fx=0.5, fy=0.5)
            small_curr = cv2.resize(gray, (0,0), fx=0.5, fy=0.5)
            shift, _ = cv2.phaseCorrelate(np.float32(small_prev), np.float32(small_curr))
            dx, dy = shift[0] * 2, shift[1] * 2 # scale back up
            
        self.prev_gray = gray
        
        # We might only run heavy CNN inference every N frames to save CPU
        # But we must update tracker shift every frame
        if self.frame_count % self.process_every_n != 0:
            # Update tracks without new detections
            confirmed_tracks = self.tracker.update([], global_shift=(dx, dy))
            return confirmed_tracks, frame
            
        # Detect
        detections = self.localizer.detect(frame)
        
        # Measure and assess
        h_frame, w_frame = frame.shape[:2]
        
        for det in detections:
            # Measure
            skel, length_px, mean_width_px, max_width_px, area_px = measure_crack(det['patch_mask'])
            det['skeleton'] = skel
            
            # Position in frame string e.g. Top-Left
            x, y, w, h = det['bbox']
            pos_y = "Top" if y < h_frame/3 else "Bottom" if y > 2*h_frame/3 else "Center"
            pos_x = "Left" if x < w_frame/3 else "Right" if x > 2*w_frame/3 else "Center"
            det['position_in_frame'] = f"{pos_y}-{pos_x}"
            
            # Assess severity
            sev_stats = self.assessor.assess(
                length_px, mean_width_px, max_width_px, area_px, det['confidence'], frame.shape
            )
            det['severity_stats'] = sev_stats
            det['timestamp'] = timestamp
            det['frame_idx'] = self.frame_count
            
        # Track
        confirmed_tracks = self.tracker.update(detections, global_shift=(dx, dy))
        
        # Database Upsert
        for track in confirmed_tracks:
            # Check if we need to save a snapshot
            # Save snapshot logic: if it's the first time it's confirmed, or if it got worse
            # For simplicity, we just pass the frame crop or full frame
            
            # Let's crop the crack
            x, y, w, h = track['bbox']
            # Safely crop
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(w_frame, int(x+w)), min(h_frame, int(y+h))
            crop = frame[y1:y2, x1:x2]
            
            if crop.size > 0:
                crop_dir = self.config['output']['crops_dir']
                os.makedirs(crop_dir, exist_ok=True)
                crop_path = os.path.join(crop_dir, f"{self.session_id}_{track['crack_id']}.jpg")
                if not os.path.exists(crop_path): # Just save first best view for now
                    cv2.imwrite(crop_path, crop)
                track['cropped_path'] = crop_path
                
            self.db.upsert_crack(self.session_id, track)
            
        return confirmed_tracks, frame
        
    def end_session(self):
        self.is_running = False
        self.db.end_session(self.session_id)

import abc
import cv2
import numpy as np
from core.preprocess import preprocess_patch

class Localizer(abc.ABC):
    @abc.abstractmethod
    def detect(self, frame) -> list:
        """
        Takes a BGR frame.
        Returns a list of dictionaries, each representing a detection:
        {
            'bbox': (x, y, w, h),
            'centroid': (cx, cy),
            'confidence': float,
            'patch_mask': np.ndarray (binary mask of the crack within the bbox),
            'contour': np.ndarray
        }
        """
        pass

class TiledClassifierLocalizer(Localizer):
    def __init__(self, model, config):
        self.model = model
        self.config = config['inference']
        self.img_size = tuple(config['model']['image_size'])
        self.window_size = self.config['window_size']
        self.stride = self.config['stride']

    def generate_heatmap(self, frame):
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        patches = []
        coords = []
        
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                patch = frame[y:y+self.window_size, x:x+self.window_size]
                processed_patch = preprocess_patch(patch, self.img_size)
                patches.append(processed_patch)
                coords.append((x, y))
                
        if not patches:
            return heatmap
            
        patches = np.array(patches)
        preds = self.model.predict(patches, batch_size=32, verbose=0).ravel()
        
        for (x, y), pred in zip(coords, preds):
            heatmap[y:y+self.window_size, x:x+self.window_size] += pred
            count_map[y:y+self.window_size, x:x+self.window_size] += 1
            
        count_map[count_map == 0] = 1
        heatmap = heatmap / count_map
        return heatmap

    def detect(self, frame):
        heatmap = self.generate_heatmap(frame)
        mask = (heatmap > self.config['heatmap_threshold']).astype(np.uint8) * 255
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            if cv2.contourArea(contour) < 50:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
                
            confidence = float(np.max(heatmap[y:y+h, x:x+w]))
            
            patch_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(patch_mask, [contour - [x, y]], -1, 255, -1)
            
            detections.append({
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'confidence': confidence,
                'patch_mask': patch_mask,
                'contour': contour,
                'heatmap': heatmap # Attach heatmap for visualization or GradCAM overlay if needed
            })
            
        return detections

class DetectorLocalizer(Localizer):
    """
    Stub for a future YOLO or Segmentation model.
    """
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def detect(self, frame):
        # TODO: Implement YOLO/Segmentation forward pass and post-processing
        # Return identical list of detection dictionaries
        return []

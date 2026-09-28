import cv2
import numpy as np
from skimage.morphology import skeletonize

class Localizer:
    def __init__(self, model, config):
        self.model = model
        self.config = config['inference']
        self.img_size = tuple(config['model']['image_size'])
        self.window_size = self.config['window_size']
        self.stride = self.config['stride']
        
    def generate_heatmap(self, frame):
        """
        Slide a window across the frame and predict 'crack' probability for each patch.
        Returns a probability heatmap of the same size as the frame.
        """
        h, w = frame.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)
        
        # Prepare batches of patches to speed up inference
        patches = []
        coords = []
        
        for y in range(0, h - self.window_size + 1, self.stride):
            for x in range(0, w - self.window_size + 1, self.stride):
                patch = frame[y:y+self.window_size, x:x+self.window_size]
                patch = cv2.resize(patch, self.img_size)
                patch = patch.astype(np.float32) / 255.0
                patches.append(patch)
                coords.append((x, y))
                
        if not patches:
            return heatmap
            
        patches = np.array(patches)
        preds = self.model.predict(patches, batch_size=32, verbose=0).ravel()
        
        for (x, y), pred in zip(coords, preds):
            heatmap[y:y+self.window_size, x:x+self.window_size] += pred
            count_map[y:y+self.window_size, x:x+self.window_size] += 1
            
        # Avoid division by zero
        count_map[count_map == 0] = 1
        heatmap = heatmap / count_map
        return heatmap

    def process_frame(self, frame):
        """
        Pipeline: Heatmap -> Threshold -> Morphological refinement -> Contours & Skeleton
        Returns: list of detections [{'id': int, 'bbox': (x,y,w,h), 'centroid': (cx, cy), 'confidence': float, 'skeleton': array, 'contour': array}]
        """
        heatmap = self.generate_heatmap(frame)
        
        # Threshold heatmap
        mask = (heatmap > self.config['heatmap_threshold']).astype(np.uint8) * 255
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < 50: # filter small noise
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate centroid
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w//2, y + h//2
                
            # Confidence is max heatmap value inside bbox
            confidence = float(np.max(heatmap[y:y+h, x:x+w]))
            
            # Skeletonize the crack inside the bbox
            patch_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(patch_mask, [contour - [x, y]], -1, 255, -1)
            skel = skeletonize(patch_mask > 0)
            
            detections.append({
                'id': i,  # Simple ID, will be tracked temporally in video pipeline
                'bbox': (x, y, w, h),
                'centroid': (cx, cy),
                'confidence': confidence,
                'contour': contour,
                'skeleton': skel,
                'patch_mask': patch_mask
            })
            
        return detections, heatmap, mask

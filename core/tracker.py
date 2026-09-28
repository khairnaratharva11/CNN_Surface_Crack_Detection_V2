import numpy as np

def compute_iou(boxA, boxB):
    # box: (x, y, w, h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-5)
    return iou

class CrackTracker:
    def __init__(self, max_age=15, iou_threshold=0.3, min_hits=3):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.min_hits = min_hits
        
        self.tracks = {} # crack_id -> {'bbox': (x,y,w,h), 'age': 0, 'hits': 1, 'data': det}
        self.next_id = 1
        
    def update(self, detections, global_shift=(0, 0)):
        """
        global_shift: (dx, dy) to compensate for camera movement
        detections: list of dicts with 'bbox' and other data
        """
        # Compensate existing tracks
        dx, dy = global_shift
        for t_id, t_data in self.tracks.items():
            x, y, w, h = t_data['bbox']
            t_data['bbox'] = (x + dx, y + dy, w, h)
            t_data['age'] += 1
            
        matched_indices = set()
        
        for det in detections:
            bbox = det['bbox']
            best_iou = self.iou_threshold
            best_id = None
            
            for t_id, t_data in self.tracks.items():
                iou = compute_iou(bbox, t_data['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_id = t_id
                    
            if best_id is not None:
                self.tracks[best_id]['bbox'] = bbox
                self.tracks[best_id]['age'] = 0
                self.tracks[best_id]['hits'] += 1
                self.tracks[best_id]['data'] = det
                matched_indices.add(best_id)
            else:
                self.tracks[self.next_id] = {
                    'bbox': bbox,
                    'age': 0,
                    'hits': 1,
                    'data': det
                }
                matched_indices.add(self.next_id)
                self.next_id += 1
                
        # Remove old tracks
        tracks_to_delete = [t_id for t_id, t_data in self.tracks.items() if t_data['age'] > self.max_age]
        for t_id in tracks_to_delete:
            del self.tracks[t_id]
            
        # Return confirmed tracks
        confirmed_tracks = []
        for t_id, t_data in self.tracks.items():
            if t_data['hits'] >= self.min_hits:
                track_info = dict(t_data['data'])
                track_info['crack_id'] = t_id
                track_info['hits'] = t_data['hits']
                confirmed_tracks.append(track_info)
                
        return confirmed_tracks

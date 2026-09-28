class SeverityAssessor:
    def __init__(self, config):
        self.config = config['severity']
        self.calib_px_per_mm = config['camera'].get('calibration_pixels_per_mm', None)
        
    def assess(self, length_px, mean_width_px, max_width_px, area_px, confidence, frame_shape):
        """
        Computes severity metrics and determines score and level.
        Returns a dictionary of severity stats.
        """
        frame_area = frame_shape[0] * frame_shape[1]
        area_ratio = float(area_px / frame_area)
        
        if self.calib_px_per_mm:
            length = length_px / self.calib_px_per_mm
            mean_width = mean_width_px / self.calib_px_per_mm
            max_width = max_width_px / self.calib_px_per_mm
            unit = 'mm'
        else:
            length = float(length_px)
            mean_width = float(mean_width_px)
            max_width = float(max_width_px)
            unit = 'px'
            
        # Simplified scoring logic (0-100)
        # width is weighted highest.
        # This is configurable, but hardcoding a simple relation for now.
        score = min(100.0, (mean_width * 5.0) + (length * 0.5) + (confidence * 20.0))
        
        # Override rules
        l_low = self.config['length_low']
        l_mod = self.config['length_moderate']
        l_high = self.config['length_high']
        
        if length >= l_high:
            severity_class = "Critical"
            score = max(score, 80.0)
        elif length >= l_mod:
            severity_class = "High"
            score = max(score, 60.0)
        elif length >= l_low:
            severity_class = "Moderate"
            score = max(score, 40.0)
        else:
            severity_class = "Low"
            
        action = self.config['actions'].get(severity_class, "Unknown")
        
        return {
            'length': length,
            'mean_width': mean_width,
            'max_width': max_width,
            'area_ratio': area_ratio,
            'unit': unit,
            'score': score,
            'severity_class': severity_class,
            'action': action,
            'is_calibrated': self.calib_px_per_mm is not None
        }

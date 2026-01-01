from boxmot import BotSort
import numpy as np
from pathlib import Path

class BoTSORTTracker:
    def __init__(self):
        # Initialize BoT-SORT tracker with stronger ReID model
        self.tracker = BotSort(
            # Use a stronger ReID model for better occlusion handling
            reid_weights=Path('clip_market1501.pt'),  # or 'osnet_x1_0_msmt17.pt'
            device='0',  
            half=False,
            track_high_thresh=0.25,    # Lowered to catch more detections initially
            track_low_thresh=0.05,     # Keep low for second-stage matching
            new_track_thresh=0.5,      # Raised to prevent spurious new tracks
            match_thresh=0.8,          # Raised for stricter appearance matching
            proximity_thresh=0.5,      # IoU threshold for spatial proximity
            appearance_thresh=0.25,    # Appearance similarity threshold
            track_buffer=60,           # Increased buffer to remember lost tracks longer
            frame_rate=14
        )

    def update(self, detections, frame):
        """
        detections:  list of [x1, y1, x2, y2, conf, cls]
        frame: current image frame (numpy array)
        returns: tracks with IDs
        """
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        tracks = self. tracker.update(dets, frame)
        return tracks

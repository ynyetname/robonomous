from boxmot import BotSort
import numpy as np
from pathlib import Path
from config_loader import load_config

class BoTSORTTracker:
    def __init__(self, config_path='config.yaml'):
        
        config = load_config(config_path='config.yaml')
        tracker_config = config['tracker']
        
        # Initializing BoT-SORT tracker with config values
        self.tracker = BotSort(      # Composition 
            reid_weights=Path(tracker_config['reid_weights']),
            device=tracker_config['device'],  
            half=tracker_config['half'],
            track_high_thresh=tracker_config['track_high_thresh'],
            track_low_thresh=tracker_config['track_low_thresh'],
            new_track_thresh=tracker_config['new_track_thresh'],
            match_thresh=tracker_config['match_thresh'],
            proximity_thresh=tracker_config['proximity_thresh'],
            appearance_thresh=tracker_config['appearance_thresh'],
            track_buffer=tracker_config['track_buffer'],
            frame_rate=tracker_config['frame_rate']
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
        tracks = self.tracker.update(dets, frame)
        return tracks



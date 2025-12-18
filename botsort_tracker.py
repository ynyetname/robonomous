from boxmot import BotSort
import numpy as np
from pathlib import Path

class BoTSORTTracker:
    def __init__(self):
        # Initialize BoT-SORT tracker
        self.tracker = BotSort(
            reid_weights=Path('osnet_x0_25_msmt17.pt'),
            device='cpu',
            half=False,
            track_high_thresh=0.5,
            track_low_thresh=0.1,
            new_track_thresh=0.6,
            match_thresh=0.8,
            frame_rate=14
        )

    def update(self, detections, frame):
        """
        detections: list of [x1, y1, x2, y2, conf, cls]
        frame: current image frame (numpy array)
        returns: tracks with IDs
        """
        if len(detections) == 0:
            return []

        dets = np.array(detections)
        tracks = self.tracker.update(dets, frame)
        return tracks
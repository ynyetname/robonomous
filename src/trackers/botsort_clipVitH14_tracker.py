from boxmot import BotSort
import numpy as np
from pathlib import Path
from torchvision import transforms
import torch

class BoTSORTTracker: 
    def __init__(self, device='0'):
        """
        Initialize BoT-SORT tracker with CLIP ViT-H/14 ReID model
        """
        self.device = device
        
        # CLIP ViT-H/14 specific preprocessing transforms
        self. reid_transforms = transforms. Compose([
            transforms.ToPILImage(),
            transforms. Resize((224, 224)),  # ViT-H/14 input size
            transforms. ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP normalization
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])
        
        # Initialize BoT-SORT tracker with CLIP ViT-H/14 ReID model
        self.tracker = BotSort(
            # CLIP ViT-H/14 weights for ReID
            reid_weights=Path('clip_market1501.pt'),
            device=device,
            half=True,  # FP16 for memory efficiency with large ViT-H/14 model
            
            # Detection thresholds
            track_high_thresh=0.25,    # Lower threshold for initial detection matching
            track_low_thresh=0.05,     # Keep low for second-stage matching
            new_track_thresh=0.5,      # Higher to prevent spurious new tracks
            
            # Matching thresholds (tuned for CLIP embeddings)
            match_thresh=0.7,          # Adjusted for CLIP similarity distribution
            proximity_thresh=0.5,      # IoU threshold for spatial proximity
            appearance_thresh=0.15,    # Lower threshold for CLIP features
            
            # Track management
            track_buffer=60,           # Frames to remember lost tracks
            frame_rate=14
        )
    
    def preprocess_frame(self, frame):
        """
        Preprocess frame for CLIP ViT-H/14 model
        
        Args:
            frame: BGR numpy array (H, W, C)
        Returns:
            Preprocessed tensor
        """
        # Convert BGR to RGB
        frame_rgb = frame[: , :, ::-1]. copy()
        return frame_rgb
    
    def update(self, detections, frame):
        """
        Update tracker with new detections
        
        Args:
            detections: list of [x1, y1, x2, y2, conf, cls]
            frame: current image frame (numpy array, BGR format)
        Returns:
            tracks:  numpy array with columns [x1, y1, x2, y2, track_id, conf, cls, idx]
        """
        if len(detections) == 0:
            # Return empty array with correct shape
            return np.empty((0, 8))
        
        # Convert detections to numpy array
        dets = np.array(detections)
        
        # Preprocess frame (BGR to RGB conversion)
        frame_processed = self.preprocess_frame(frame)
        
        # Update tracker
        tracks = self. tracker.update(dets, frame_processed)
        
        return tracks
    
    def reset(self):
        """
        Reset the tracker state
        """
        self.tracker.reset()


# Example usage
if __name__ == "__main__": 
    import cv2
    
    # Initialize tracker
    tracker = BoTSORTTracker(device='0')
    
    # Example:  Process video
    video_path = "input_video.mp4"
    cap = cv2.VideoCapture(video_path)
    
    while True: 
        ret, frame = cap.read()
        if not ret: 
            break
        
        # Your detection model here (e.g., YOLO)
        # detections format: [[x1, y1, x2, y2, confidence, class_id], ...]
        detections = []  # Replace with your detector output
        
        # Update tracker
        tracks = tracker.update(detections, frame)
        
        # Process tracks
        for track in tracks:
            x1, y1, x2, y2, track_id, conf, cls, idx = track
            
            # Draw bounding box
            cv2.rectangle(
                frame, 
                (int(x1), int(y1)), 
                (int(x2), int(y2)), 
                (0, 255, 0), 
                2
            )
            
            # Draw track ID
            cv2.putText(
                frame,
                f"ID: {int(track_id)}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
        
        # Display frame
        cv2.imshow("BoT-SORT with CLIP ViT-H/14", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

import cv2
import time
import numpy as np
from ultralytics import YOLO
from botsort_clipVitb16_tracker import BoTSORTTracker
from config_loader import load_config

config = load_config(config_path="config.yaml")

detection_config = config['detection']
video_config = config['video']

model = YOLO(detection_config['model'])
tracker = BoTSORTTracker()

cap = cv2.VideoCapture(video_config['video_file_path'])

# Setup video writer to save output
fps = cap.get(cv2.CAP_PROP_FPS)
# Use source FPS if valid, otherwise default to 30 FPS
if fps <= 0 or fps > 120:
    fps = 30
else:
    fps = fps / 4  # Slow down the output video by quarter

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(video_config['robotics_output_name'], fourcc, fps, (width, height))
print(f"Output video FPS: {fps}")

# Dictionary to store unique colors for each track ID
id_colors = {}

def get_color_for_id(track_id):
    """Get a consistent color for a given track ID."""
    if track_id not in id_colors:
        # Generate a random bright color (avoid too dark colors)
        color = tuple(np.random.randint(100, 255, 3).tolist())
        id_colors[track_id] = color
    return id_colors[track_id]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=detection_config['confidence_threshold'], classes=detection_config['target_classes'])
    
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())
        detections.append([float(x1), float(y1), float(x2), float(y2), float(conf), cls])

    tracks = tracker.update(detections, frame)

    robot_output = []

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        w = x2 - x1
        h = y2 - y1
        
        robot_output.append({
            "id": track_id,
            "bbox": [x1, y1, w, h],
            "timestamp": time.time()
        })

        # Get unique color for this track ID
        color = get_color_for_id(track_id)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    out.write(frame)  # Save frame to output video
    print(robot_output)
    cv2.imshow("Robonomous-Robotics Demo", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):  # ESC or 'q' to quit
        break

cap.release()
out.release()  # Release the video writer
cv2.destroyAllWindows()

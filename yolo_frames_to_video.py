import cv2
import os
import numpy as np
from ultralytics import YOLO
from botsort_clipVitb16_tracker import BoTSORTTracker
from config_loader import load_config

config = load_config(config_path='config.yaml')

detect_config = config['detection']
path_config = config['paths']
tracker_config = config['tracker']
video_config = config['video']

frames_path = path_config['frames_path']
output_video = video_config['output_video_name']
model_path = "yolov8m.pt"
pred_file = open("predictions.txt", "w")

confidence_threshold = detect_config['confidence_threshold']
target_classes = detect_config['target_classes']  
fps = tracker_config['frame_rate']   

print("BoT-SORT Tracker with CLIP ReID")         

print("\n[1/2] Loading YOLOv8 model")
model = YOLO(model_path)
print(f" Loaded: {model_path}")

print("\n[2/2] Initializing BoT-SORT tracker")
tracker = BoTSORTTracker(config_path='config.yaml')    
print("Tracker initialized")

frame_files = sorted([
    f for f in os.listdir(frames_path)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

total_frames = len(frame_files)
if total_frames == 0:
    raise RuntimeError(f"No frames found in {frames_path}")

first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
if first_frame is None:
    raise RuntimeError("Could not read first frame")

height, width = first_frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
if not video.isOpened():
    raise RuntimeError("Could not create output video")

print(f"Input:{frames_path}")
print(f"Output:{output_video}")
print(f"Total Frames:{total_frames}")
print(f"Resolution:{width}x{height} @ {fps}fps")
print(f"All classes")

track_colors = {}

print("Processing frames")

for i, frame_name in enumerate(frame_files):
    frame_path = os.path.join(frames_path, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"Warning: Could not read {frame_name}, skipping.")
        continue

    results = model(
        frame,
        conf=confidence_threshold,
        classes=target_classes,
        verbose=False
    )

    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0].cpu().numpy())
        cls = int(box.cls[0].cpu().numpy())

        detections.append([
            float(x1), float(y1), float(x2), float(y2),
            float(conf), cls
        ])

    tracks = tracker.update(detections, frame)

    if i == 0:  # Debug first frame
        print(f"Frame 1: YOLO detections={len(detections)}, Tracker outputs={len(tracks)}")

    # Write tracker outputs (stable IDs) instead of raw detections
    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])
        w = x2 - x1
        h = y2 - y1
        frame_id = i + 1
        pred_file.write(f"{frame_id},{track_id},{x1},{y1},{w},{h}\n")

    annotated_frame = frame.copy()

    for track in tracks:
        x1, y1, x2, y2 = map(int, track[:4])
        track_id = int(track[4])

        if track_id not in track_colors:
            np.random.seed(track_id)
            track_colors[track_id] = tuple(
                int(c) for c in np.random.randint(50, 255, 3)
            )
        color = track_colors[track_id]

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{track_id}"
        (lw, lh), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            annotated_frame,
            (x1, y1 - lh - 10),
            (x1 + lw + 5, y1),
            color,
            -1
        )

        cv2.putText(
            annotated_frame,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    info_text = (
        f"Frame:{i+1}/{total_frames} | "
        f"Detections:{len(detections)} | "
        f"Tracks:{len(tracks)}"
    )

    cv2.putText(
        annotated_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    video.write(annotated_frame)

    if (i + 1) % 50 == 0 or (i + 1) == total_frames:
        progress = (i + 1) / total_frames * 100
        print(f"[{i+1:5d}/{total_frames}] {progress:5.1f}% | Active Tracks: {len(tracks)}")

video.release()
pred_file.close()

print(f"Video saved:{output_video}")
print(f"Frames processed:{total_frames}")
print(f"Unique tracks:{len(track_colors)}")

def get_color_for_id(track_id):
    """
    Generating a unique color for each track ID
    """
    np.random.seed(track_id)
    color = tuple(map(int, np.random.randint(0, 255, 3)))
    return color

import cv2
import os
import numpy as np
from ultralytics import YOLO
from botsort_clipVitb16_tracker import BoTSORTTracker

# ------------------- PATHS -------------------
frames_path = r"C:\Users\Ayyan Aftab\Downloads\MOT17\MOT17\train\MOT17-04-FRCNN\img1"
output_video = "mot17_clipVitB16(1)_tracked.mp4"
model_path = "yolov8m.pt"

# ------------------- SETTINGS -------------------
confidence_threshold = 0.5
target_classes = [0]  # person class only
fps = 30

print("=" * 60)
print("BoT-SORT Tracker with CLIP ReID (GPU FP32)")
print("=" * 60)

# ------------------- LOAD YOLO -------------------
print("\n[1/2] Loading YOLOv8 model...")
model = YOLO(model_path)
print(f"      ✓ Loaded: {model_path}")

# ------------------- INIT TRACKER -------------------
print("\n[2/2] Initializing BoT-SORT tracker...")
tracker = BoTSORTTracker(device='0')  # GPU FP32
print("      ✓ Tracker initialized")

# ------------------- LOAD FRAMES -------------------
frame_files = sorted([
    f for f in os.listdir(frames_path)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])

total_frames = len(frame_files)
if total_frames == 0:
    raise RuntimeError(f"No frames found in {frames_path}")

# Read first frame for video info
first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
if first_frame is None:
    raise RuntimeError("Could not read first frame")

height, width = first_frame.shape[:2]

# ------------------- VIDEO WRITER -------------------
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
if not video.isOpened():
    raise RuntimeError("Could not create output video")

print("\n" + "-" * 60)
print(f"Input:         {frames_path}")
print(f"Output:        {output_video}")
print(f"Total Frames:  {total_frames}")
print(f"Resolution:    {width}x{height} @ {fps}fps")
print(f"Target Class:  Person (0)")
print("-" * 60 + "\n")

# ------------------- TRACK COLORS -------------------
track_colors = {}

print("Processing frames...")

# =================== MAIN LOOP ===================
for i, frame_name in enumerate(frame_files):
    frame_path = os.path.join(frames_path, frame_name)
    frame = cv2.imread(frame_path)

    if frame is None:
        print(f"⚠ Warning: Could not read {frame_name}, skipping...")
        continue

    # ---------- YOLO DETECTION ----------
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

    # ---------- TRACKING ----------
    tracks = tracker.update(detections, frame)

    # ---------- DRAW ----------
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

    # ---------- INFO ----------
    info_text = (
        f"Frame: {i+1}/{total_frames} | "
        f"Detections: {len(detections)} | "
        f"Tracks: {len(tracks)}"
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

    # ---------- WRITE ----------
    video.write(annotated_frame)

    if (i + 1) % 50 == 0 or (i + 1) == total_frames:
        progress = (i + 1) / total_frames * 100
        print(f"[{i+1:5d}/{total_frames}] {progress:5.1f}% | Active Tracks: {len(tracks)}")

# =================== CLEANUP ===================
video.release()

print("\n" + "=" * 60)
print("COMPLETE!")
print("=" * 60)
print(f"✓ Video saved:        {output_video}")
print(f"✓ Frames processed:  {total_frames}")
print(f"✓ Unique tracks:     {len(track_colors)}")
print("=" * 60)

def get_color_for_id(track_id):
    """
    Generate a unique color for each track ID
    """
    np.random.seed(track_id)
    color = tuple(map(int, np.random. randint(0, 255, 3)))
    return color
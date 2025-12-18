import cv2
import os
from ultralytics import YOLO
from botsort_tracker import BoTSORTTracker

# CHANGING THE PATH to the img1 folder
frames_path = r"C:\Users\Ayyan Aftab\Downloads\MOT17\MOT17\train\MOT17-04-FRCNN\img1"
output_video = "mot17_04_detected.mp4"  # Output video with detections
model_path = "yolov8m.pt"               # YOLOv8m model 
confidence_threshold = 0.5              # Detection confidence threshold

# Loading YOLOv8 model
print("Loading YOLOv8 model...") 
model = YOLO(model_path)

# Initialize tracker
tracker = BoTSORTTracker()

# Receiving list of frame files
frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith(".jpg")])
total_frames = len(frame_files)

# Read first frame to get size
first_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
height, width, _ = first_frame.shape

# MOT17 fps (check seqinfo.ini, usually 30)
fps = 14

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

print(f"Processing {total_frames} frames...")
print(f"Frame size: {width}x{height} @ {fps}fps")

# Processing each frame
for i, frame_name in enumerate(frame_files):
    frame_path = os.path.join(frames_path, frame_name)
    frame = cv2.imread(frame_path)
    
    # Running YOLOv8 detection on the frame
    results = model(frame, conf=confidence_threshold, verbose=False)
    
    # Extract detections [x1, y1, x2, y2, conf, cls]
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = box.conf[0].cpu().numpy()
        cls = box.cls[0].cpu().numpy()
        detections.append([x1, y1, x2, y2, conf, cls])
    
    # Update tracker
    tracks = tracker.update(detections, frame)
    
    # Draw tracked objects with IDs
    annotated_frame = frame.copy()
    for track in tracks:
        x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'ID: {track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Writing the annotated frame to video
    video.write(annotated_frame)
    
    # Print progress every 50 frames
    if (i + 1) % 50 == 0 or (i + 1) == total_frames:
        print(f"Processed {i + 1}/{total_frames} frames ({((i + 1)/total_frames)*100:.1f}%)")

video.release()
print(f"\n✓ Video saved as: {output_video}")
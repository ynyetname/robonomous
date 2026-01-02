#  Robonomous

<!-- Badges -->
![Python](https://img.shields.io/badge/Python-3.9+-blue. svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-8.3.233-green.svg)
![BoT-SORT](https://img.shields.io/badge/Tracker-BoT--SORT-red.svg)
![CLIP](https://img.shields.io/badge/ReID-CLIP-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

<!-- Performance Badges -->
![HOTA](https://img.shields.io/badge/HOTA-43.3%25-brightgreen.svg)
![MOTA](https://img.shields.io/badge/MOTA-37.0%25-yellow.svg)
![IDF1](https://img.shields.io/badge/IDF1-47.2%25-blue.svg)

---

**Multi-Object Tracking (MOT) with YOLOv8 Detection and BoT-SORT Tracking**

Real-time object detection and tracking system using YOLOv8 for detection and BoT-SORT with CLIP-based ReID for robust multi-object tracking.

![Detection Demo](assets/detection_demo. jpg)

---

##  Features

- **YOLOv8 Detection** — Fast and accurate object detection
- **BoT-SORT Tracking** — State-of-the-art multi-object tracking
- **ReID Integration** — Re-identification for handling occlusions
- **CLIP-based ReID** — Using CLIP ViT models for robust appearance matching
- **Real-time Processing** — Live webcam and video file support
- **MOT17 Evaluation** — Built-in evaluation scripts for benchmarking

---

##  Demo

### Detection Results

![YOLOv8 Detection](assets/detection_demo.jpg)

### Full Video Demo

(https://github.com/user-attachments/assets/f76395c7-b3ff-482d-a41a-e095012ac4a2)
> *Video showing real-time detection + tracking with ReID*

---

##  How It Works

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│             │    │             │    │             │    │             │
│ Input Video │───▶│   YOLOv8    │───▶│  BoT-SORT   │───▶│   Output    │
│             │    │  Detection  │    │  Tracking   │    │   Video     │
│             │    │             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                          │                  │
                          ▼                  ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Bounding   │    │    CLIP     │
                   │   Boxes     │    │    ReID     │
                   │ + Confidence│    │  Matching   │
                   └─────────────┘    └─────────────┘
```

### Pipeline Steps

| Step | Component | Description |
|------|-----------|-------------|
| 1️) | **Input** | Video file or webcam stream |
| 2️) | **Detection** | YOLOv8 detects objects (persons) |
| 3️) | **Tracking** | BoT-SORT assigns track IDs |
| 4️) | **ReID** | CLIP model handles occlusions |
| 5️) | **Output** | Video with tracked bounding boxes |

---

##  Evaluation Results

### MOT17-04-FRCNN Benchmark

**Tracker:** BoT-SORT + CLIP ViT-B/16

| Metric | Score | Description |
|--------|-------|-------------|
| **HOTA** | 43.3% | Higher Order Tracking Accuracy |
| **MOTA** | 37.0% | Multi-Object Tracking Accuracy |
| **IDF1** | 47.2% | ID F1 Score |
| **Precision** | 93.9% | Correct detections / Total predictions |
| **Recall** | 39.9% | Correct detections / Total ground truth |

### Detailed Metrics

| Metric | Value |
|--------|-------|
| ID Switches (IDSW) | 364 |
| False Positives (FP) | 2,815 |
| False Negatives (FN) | 64,861 |

### ID Statistics

| Type | Count |
|------|-------|
| Ground Truth IDs | 141 |
| Predicted IDs | 255 |

### Key Observations

-  **High Precision (93.9%)** — Very few false detections
-  **Low Recall (39.9%)** — Many objects missed (due to occlusion/crowd)
-  **ID Switches (364)** — Some ID swapping during occlusions

---

##  Project Structure

```
robonomous/
├── assets/                  # Demo images and videos
│   ├── detection_demo. jpg
│   └── tracking_demo.gif
├── configs/                 # Configuration files
│   └── configs.yaml
├── configs_loader/          # Config loading utilities
│   └── config_loader.py
├── demos/                   # Demo scripts
│   └── robotics_demo.py
├── evaluation/              # Evaluation scripts
│   ├── evaluation_mot17.py
│   └── hota_evaluation.py
├── predictions/             # Model predictions
│   ├── predictions.txt
│   └── predictions_cleaned.txt
├── src/                     # Source code
│   └── trackers/
│       ├── __init__.py
│       ├── botsort_tracker.py
│       ├── botsort_clipVitb16_tracker.py
│       └── botsort_clipVitH14_tracker.py
├── utils/                   # Utility functions
│   ├── data_cleaning.py
│   └── yolo_frames_to_video.py
├── . gitignore
├── environment.yaml
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

---

##  Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/ynyetname/robonomous.git
cd robonomous

# Create conda environment
conda env create -f environment.yaml

# Activate environment
conda activate mot_env
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/ynyetname/robonomous.git
cd robonomous

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

---

##  Dependencies

| Package | Version | Description |

| ultralytics | 8.3.233 | YOLOv8 implementation |
| boxmot | latest | BoT-SORT tracker |
| opencv-python | 4.12.0.88 | Image processing |
| numpy | 2.2.6 | Numerical computing |
| torch | latest | Deep learning framework |
| torchvision | latest | Vision utilities |

---

##  Quick Start

### 1. Run on Video File

Edit `configs/configs.yaml`:

### 2. Run Real-time Demo (Webcam)

```bash
cd demos
python robotics_demo.py
```

```yaml
video: 
"C:\Users\Ayyan Aftab\Desktop\BYOP\mot17_clipVitB16(0)_tracked.mp4"

Then run:

```bash
python demos/robotics_demo.py
```

### 3. Run MOT17 Evaluation

```bash
python evaluation/evaluation_mot17.py
python evaluation/hota_evaluation.py
```

---

##  Configuration

Edit `configs/configs.yaml` to customize:

```yaml
# Detection Settings
detection:
  model:  "yolov8m. pt"           # YOLOv8 model (n/s/m/l/x)
  confidence_threshold: 0.01    # Detection confidence
  target_classes: [0]           # 0 = person

# Tracker Settings
tracker:
  reid_weights: "clip_market1501.pt"  # ReID model
  track_high_thresh: 0.05
  track_low_thresh: 0.01
  match_thresh: 0.8
  appearance_thresh: 0.15
  track_buffer: 1200

# Video Settings
video:
  fps: 30
  codec: "mp4v"
  video_file_path: 0            # 0 for webcam, or path to video
```

---

##  Supported ReID Models

| Model | Description | Performance |

| `clip_market1501.pt` | CLIP ViT-B/16 trained on Market1501 | Best |
| `osnet_x1_0_msmt17.pt` | OSNet trained on MSMT17 | Good |
| `osnet_x0_25_msmt17.pt` | Lightweight OSNet | Fast |

---

##  Available Trackers

| Tracker | File | Description |

| **BoT-SORT** | `botsort_tracker.py` | Basic tracker |
| **BoT-SORT + CLIP ViT-B/16** | `botsort_clipVitb16_tracker. py` | Enhanced with CLIP |
| **BoT-SORT + CLIP ViT-H/14** | `botsort_clipVitH14_tracker.py` | High-performance |

---

##  Evaluation Metrics Explained

| Metric | What It Measures | Good Score |
|--------|------------------|------------|
| **HOTA** | Overall tracking accuracy | > 50% |
| **MOTA** | Detection + tracking accuracy | > 50% |
| **IDF1** | How well IDs are maintained | > 55% |
| **Precision** | False positive rate | > 90% |
| **Recall** | False negative rate | > 70% |

---

##  Output Format

Tracking output follows MOT format:

```
frame_id, track_id, x, y, width, height, confidence, -1, -1, -1
```

Example:
```
1, 1, 100, 200, 50, 120, 0.95, -1, -1, -1
1, 2, 300, 150, 45, 110, 0.89, -1, -1, -1
2, 1, 102, 198, 50, 120, 0.94, -1, -1, -1
```

---

##  Future Improvements

- [ ] Improve tracking accuracy during occlusions and camera angle changes
- [ ] Enhance real-time webcam detection and tracking performance
- [ ] Add YOLOv9/YOLOv10 support
- [ ] Implement DeepSORT tracker for comparison
- [ ] Support for multiple camera inputs
- [ ] Docker container for easy deployment
- [ ] Add more ReID models (OSNet, MGN)
- [ ] Improve recall with better detection confidence tuning

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

##  License

This project is open source and available under the [MIT License](LICENSE).

---

##  Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — Object detection
- [BoxMOT](https://github.com/mikel-brostrom/boxmot) — BoT-SORT implementation
- [MOT17 Dataset](https://motchallenge.net/data/MOT17/) — Benchmark dataset
- [CLIP](https://github.com/openai/CLIP) — ReID feature extraction

---

##  Contact

**Author:** ynyetname

**GitHub:** [@ynyetname](https://github.com/ynyetname)

---


```

import numpy as np
import motmetrics as mm
from config_loader import load_config

config = load_config(config_path='config.yaml')
path_config = config['paths']

gtfile_path = path_config['gtfile_path']
predictions_cleaned_path = path_config['predictions_cleaned_path']

IOU_THRESHOLD = 0.5

gt = np.loadtxt(gtfile_path, delimiter=",")
pred = np.loadtxt(predictions_cleaned_path, delimiter=",")

# using only first 6 columns from gt
gt = gt[:, :6]

# initializing accumulator
acc = mm.MOTAccumulator(auto_id=True)

# frame by frame evaluation
frames = np.unique(gt[:, 0]).astype(int)

for frame in frames:
    gt_frame = gt[gt[:, 0] == frame]
    pred_frame = pred[pred[:, 0] == frame]

    gt_ids = gt_frame[:, 1].astype(int)
    pred_ids = pred_frame[:, 1].astype(int)

    gt_boxes = gt_frame[:, 2:6].copy()
    pred_boxes = pred_frame[:, 2:6].copy()

    # Converting (x,y,w,h) → (x1,y1,x2,y2)
    gt_boxes[:, 2] += gt_boxes[:, 0]
    gt_boxes[:, 3] += gt_boxes[:, 1]

    pred_boxes[:, 2] += pred_boxes[:, 0]
    pred_boxes[:, 3] += pred_boxes[:, 1]

    # IoU distance matrix
    distances = mm.distances.iou_matrix(
        gt_boxes,
        pred_boxes,
        max_iou=IOU_THRESHOLD
    )

    acc.update(
        gt_ids.tolist(),
        pred_ids.tolist(),
        distances
    )
    
# Computing metrics
mh = mm.metrics.create()

summary = mh.compute(
    acc,
    metrics=[
        "mota",
        "idf1",
        "precision",
        "recall",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "num_unique_objects" 
    ],
    name="MOT17-04-FRCNN"
)

# Counting unique predicted IDs
unique_pred_ids = len(np.unique(pred[:, 1]))
unique_gt_ids = len(np.unique(gt[:, 1]))

print(f"\nTotal Unique IDs - Ground Truth: {unique_gt_ids}, Predicted: {unique_pred_ids}")

print(
    mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap={
            "mota": "MOTA",
            "idf1": "IDF1",
            "precision": "Precision",
            "recall": "Recall",
            "num_switches": "IDSW",
            "num_false_positives": "FP",
            "num_misses": "FN",
            "num_unique_objects": "GT_IDs"
        }
    )
)

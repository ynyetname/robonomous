import numpy as np
from config_loader import load_config

config = load_config(config_path='config.yaml')
path_config = config['paths']

gtfile_path = path_config['gtfile_path']
predictions_cleaned_path = path_config['predictions_cleaned_path']

gt = np.loadtxt(gtfile_path, delimiter=",")
pred = np.loadtxt(predictions_cleaned_path, delimiter=",")

gt = gt[:, :6]

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

frames = np.unique(gt[:, 0]).astype(int)
alpha_values = np.arange(0.05, 1.0, 0.05)

hota_scores = []

for alpha in alpha_values:
    tp_sum = 0
    fp_sum = 0
    fn_sum = 0
    
    for frame in frames:
        gt_frame = gt[gt[:, 0] == frame]
        pred_frame = pred[pred[:, 0] == frame]
        
        gt_boxes = gt_frame[:, 2:6].copy()
        pred_boxes = pred_frame[:, 2:6].copy()
        
        gt_boxes[:, 2] += gt_boxes[:, 0]
        gt_boxes[:, 3] += gt_boxes[:, 1]
        pred_boxes[:, 2] += pred_boxes[:, 0]
        pred_boxes[:, 3] += pred_boxes[:, 1]
        
        matched_gt = set()
        matched_pred = set()
        
        for i, pred_box in enumerate(pred_boxes):
            for j, gt_box in enumerate(gt_boxes):
                if j not in matched_gt:
                    iou = calculate_iou(pred_box, gt_box)
                    if iou >= alpha:
                        matched_gt.add(j)
                        matched_pred.add(i)
                        tp_sum += 1
                        break
        
        fp_sum += len(pred_boxes) - len(matched_pred)
        fn_sum += len(gt_boxes) - len(matched_gt)
    
    if tp_sum + fp_sum + fn_sum > 0:
        hota_alpha = tp_sum / np.sqrt((tp_sum + fp_sum) * (tp_sum + fn_sum))
        hota_scores.append(hota_alpha)

hota = np.mean(hota_scores)

print(f"HOTA: {hota:.1%}")

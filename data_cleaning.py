import numpy as np
from collections import defaultdict

RAW_FILE = "predictions.txt"
CLEAN_FILE = "predictions_cleaned.txt"

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

MIN_TRACK_LENGTH = 5  

data = np.loadtxt(RAW_FILE, delimiter=",")

# 1) REMOVE INVALID BOXES
valid = []
for row in data:
    frame, tid, x, y, w, h = row

    if w <= 0 or h <= 0:
        continue
    if x < 0 or y < 0:
        continue
    if x + w > IMAGE_WIDTH or y + h > IMAGE_HEIGHT:
        continue

    valid.append(row)

valid = np.array(valid)

# 2) REMOVE DUPLICATE IDS PER FRAME
frame_id_map = defaultdict(list)

for row in valid:
    frame_id_map[(int(row[0]), int(row[1]))].append(row)

unique = []
for _, rows in frame_id_map.items():
    if len(rows) == 1:
        unique.append(rows[0])
    else:
        # keep box with largest area
        areas = [r[4] * r[5] for r in rows]
        unique.append(rows[np.argmax(areas)])

unique = np.array(unique)

# 3) REMOVE SHORT TRACKS - SKIP THIS STEP for temporary IDs
# Since we're using unique temp IDs per detection, skip track length filtering
final = unique

# 4) SORT BY FRAME, ID
if len(final) > 0:
    final = final[np.lexsort((final[:, 1], final[:, 0]))]

with open(CLEAN_FILE, "w") as f:
    for row in final:
        f.write(
            f"{int(row[0])},{int(row[1])},"
            f"{int(row[2])},{int(row[3])},"
            f"{int(row[4])},{int(row[5])}\n"
        )

if len(final) > 0:
    print(f"Total detections kept: {len(final)}")
    print(f"Total tracks kept: {len(set(final[:,1]))}")
else:
    print("No detections kept after filtering")

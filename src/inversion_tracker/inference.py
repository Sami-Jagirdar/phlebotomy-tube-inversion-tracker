"""Pose-model inference on whole videos -> one row of keypoints per frame.

Frames are read sequentially (real timestamps, VFR-safe) and cropped exactly as in training.
Coordinates are in crop pixels, like the labels; use Crop.to_full to map them back to the source frame.
Frames without a detection are kept as rows of NaN, so the time axis has no holes.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO

from inversion_tracker.video import Crop, iter_frames

KPT_COLUMNS = ["cap_x", "cap_y", "cap_conf", "base_x", "base_y", "base_conf"]
BOX_COLUMNS = ["box_conf", "box_x1", "box_y1", "box_x2", "box_y2"]
COLUMNS = ["frame_idx", "t_ms", *KPT_COLUMNS, *BOX_COLUMNS]


def _result_row(result) -> list[float]:
    """Keypoints + box of the highest-confidence detection (one tube per frame), or NaNs."""
    if not len(result.boxes):
        return [np.nan] * (len(KPT_COLUMNS) + len(BOX_COLUMNS))
    i = int(result.boxes.conf.argmax())
    (cx, cy), (bx, by) = result.keypoints.xy[i].tolist()
    confs = result.keypoints.conf
    c_conf, b_conf = confs[i].tolist() if confs is not None else (np.nan, np.nan)
    return [cx, cy, c_conf, bx, by, b_conf, float(result.boxes.conf[i]), *result.boxes.xyxy[i].tolist()]


def predict_video(model: YOLO, path: Path, crop: Crop, imgsz: int = 640, conf: float = 0.1,
                  batch: int = 32) -> pd.DataFrame:
    """Run the model on every frame of one video; returns a DataFrame with COLUMNS."""
    rows, pending = [], []  # pending: (frame_idx, t_ms, frame) waiting for a batched predict

    def flush() -> None:
        results = model.predict([f for _, _, f in pending], imgsz=imgsz, conf=conf, verbose=False)
        rows.extend([idx, t_ms, *_result_row(r)] for (idx, t_ms, _), r in zip(pending, results))
        pending.clear()

    for idx, t_ms, frame in iter_frames(path, crop):
        pending.append((idx, t_ms, np.ascontiguousarray(frame)))  # the crop is a view
        if len(pending) == batch:
            flush()
    if pending:
        flush()

    df = pd.DataFrame(rows, columns=COLUMNS)
    df["frame_idx"] = df["frame_idx"].astype("int32")
    float_cols = COLUMNS[2:]
    df[float_cols] = df[float_cols].astype("float32")
    return df

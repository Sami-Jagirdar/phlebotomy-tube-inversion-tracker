"""Frame selection and extraction for keypoint labelling.

Per video: score frame-to-frame motion inside the crop, then pick
  - n_motion high-motion frames (mid-inversion, blurred, occluded -- the hard cases), and
  - n_calm low-motion frames from the start/end (tube in or leaving the container, at rest),
all at least min_gap apart in time. Picked frames are saved as cropped JPEGs.

Frames are always addressed by their index in a *sequential* decode (iter_frames), never by
seeking: seeking by frame number is unreliable on variable-frame-rate video, and every later
stage (pre-labelling, inference) also decodes sequentially, so indices stay consistent.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from inversion_tracker.video import Crop, iter_frames

MOTION_WIDTH = 160      # frames are downscaled to this width before differencing
CALM_WINDOW = 0.10      # calm frames come from the first / last 10% of the video
JPEG_QUALITY = 95


def frame_name(video_id: int, frame_idx: int) -> str:
    return f"{video_id}_{frame_idx:05d}.jpg"


def motion_scores(path: Path, crop: Crop) -> pd.DataFrame:
    """Mean absolute frame difference per frame (first frame = 0). Columns: frame_idx, t_ms, motion."""
    rows, prev = [], None
    for idx, t_ms, frame in iter_frames(path, crop):
        h = int(frame.shape[0] * MOTION_WIDTH / frame.shape[1])
        small = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (MOTION_WIDTH, h)), cv2.COLOR_BGR2GRAY), (5, 5), 0)
        motion = 0.0 if prev is None else float(cv2.absdiff(small, prev).mean())
        rows.append((idx, t_ms, motion))
        prev = small
    return pd.DataFrame(rows, columns=["frame_idx", "t_ms", "motion"])


def select_frames(scores: pd.DataFrame, n_motion: int, n_calm: int, min_gap_ms: float) -> pd.DataFrame:
    """Pick calm frames first (alternating start/end windows), then the highest-motion frames,
    keeping every pick at least min_gap_ms from all others. Adds a `kind` column."""
    n = len(scores)
    k = max(1, int(n * CALM_WINDOW))
    windows = [scores.iloc[:k], scores.iloc[n - k:]]
    picked: list[tuple[int, str]] = []  # (row position, kind)

    def far_enough(t: float) -> bool:
        return all(abs(t - scores.t_ms.iat[p]) >= min_gap_ms for p, _ in picked)

    for i in range(n_calm):
        for pos in windows[i % 2].motion.sort_values().index:
            if far_enough(scores.t_ms.at[pos]):
                picked.append((scores.index.get_loc(pos), "calm"))
                break

    n_calm_picked = len(picked)
    for pos in scores.motion.sort_values(ascending=False).index:
        if len(picked) - n_calm_picked >= n_motion:
            break
        if far_enough(scores.t_ms.at[pos]):
            picked.append((scores.index.get_loc(pos), "motion"))

    out = scores.iloc[[p for p, _ in picked]].copy()
    out["kind"] = [kind for _, kind in picked]
    return out.sort_values("frame_idx").reset_index(drop=True)


def save_frames(path: Path, crop: Crop, video_id: int, selected: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Second sequential pass: write the selected cropped frames. Returns `selected` plus `file` and
    `t_ms_check` (timestamp seen on this pass; should equal t_ms)."""
    wanted = dict(zip(selected.frame_idx, range(len(selected))))
    files, t_check = [None] * len(selected), [np.nan] * len(selected)
    last = int(selected.frame_idx.max())
    for idx, t_ms, frame in iter_frames(path, crop, stop=last + 1):
        if idx in wanted:
            i = wanted[idx]
            name = frame_name(video_id, idx)
            cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            files[i], t_check[i] = name, t_ms
    out = selected.copy()
    out["file"], out["t_ms_check"] = files, t_check
    return out


def extract_video(row: pd.Series, video_file: Path, crop: Crop, out_dir: Path,
                  n_motion: int, n_calm: int, min_gap_s: float) -> pd.DataFrame:
    """Score, select and save frames for one index row. Returns its manifest rows."""
    scores = motion_scores(video_file, crop)
    selected = select_frames(scores, n_motion, n_calm, min_gap_s * 1000)
    saved = save_frames(video_file, crop, int(row.video_id), selected, out_dir)
    saved.insert(0, "video_id", int(row.video_id))
    saved.insert(1, "split", row.split)
    return saved

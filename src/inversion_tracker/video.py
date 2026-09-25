"""Video reading shared by every stage: fixed crop + real per-frame timestamps.

The videos are variable-frame-rate, so timing always comes from CAP_PROP_POS_MSEC,
never from frame_idx / fps.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml

from inversion_tracker.config import CONFIG_DIR, load_paths

CROP_FILE = CONFIG_DIR / "crop.yaml"


@dataclass(frozen=True)
class Crop:
    """Axis-aligned crop box in original-frame pixel coordinates."""

    x: int
    y: int
    w: int
    h: int

    def apply(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y:self.y + self.h, self.x:self.x + self.w]

    def to_full(self, pts: np.ndarray) -> np.ndarray:
        """Map (..., 2) points from crop coordinates back to original-frame coordinates."""
        return pts + np.array([self.x, self.y], dtype=pts.dtype)


def load_crop() -> Crop:
    if not CROP_FILE.exists():
        raise FileNotFoundError(f"{CROP_FILE} not found -- run tools/select_crop.py first.")
    cfg = yaml.safe_load(CROP_FILE.read_text())
    return Crop(cfg["x"], cfg["y"], cfg["w"], cfg["h"])


def save_crop(crop: Crop, source_size: tuple[int, int]) -> None:
    CROP_FILE.write_text(
        "# Fixed crop box in original-frame pixels (made by tools/select_crop.py).\n"
        "# Keypoint labels are in crop coordinates: changing this invalidates existing labels.\n"
        + yaml.safe_dump({"source_size": list(source_size), "x": crop.x, "y": crop.y, "w": crop.w, "h": crop.h},
                         sort_keys=False)
    )


def video_path(rel_path: str) -> Path:
    """Absolute path of a video from its `rel_path` column in the index."""
    return load_paths()["video_root"] / rel_path


def iter_frames(path: Path, crop: Crop | None = None, start: int = 0, stop: int | None = None,
                step: int = 1) -> Iterator[tuple[int, float, np.ndarray]]:
    """Yield (frame_idx, t_ms, frame) sequentially, optionally cropped.

    Skipped frames are grabbed but not decoded, so step > 1 is cheap.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open {path}")
    try:
        if start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while stop is None or idx < stop:
            if (idx - start) % step:
                if not cap.grab():
                    break
            else:
                ok, frame = cap.read()
                if not ok:
                    break
                t_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                yield idx, t_ms, crop.apply(frame) if crop else frame
            idx += 1
    finally:
        cap.release()


def read_frame(path: Path, frame_idx: int, crop: Crop | None = None) -> tuple[float, np.ndarray]:
    """Random access to a single frame. Returns (t_ms, frame). Use iter_frames for sequential reads."""
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            raise OSError(f"Cannot read frame {frame_idx} of {path}")
        return cap.get(cv2.CAP_PROP_POS_MSEC), crop.apply(frame) if crop else frame
    finally:
        cap.release()

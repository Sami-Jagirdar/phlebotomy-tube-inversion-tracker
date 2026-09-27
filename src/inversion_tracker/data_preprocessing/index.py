"""Video index and the frozen dataset split.

Walks the phlebotomy video folders, parses labels from folder/file names,
reads video metadata, runs sanity checks, and assigns each video to one split:

    kp_train / kp_val  -> videos whose frames get keypoint-labelled (YOLO train/val)
    dev                -> tunes the signal + state-machine parameters
    test               -> touched once, at the end, for the reported numbers

Splits are stratified by count bin within each tube folder, with a fixed seed.
"""

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from inversion_tracker.config import load_paths

FILENAME_RE = re.compile(r"^(\d+)_(\d+)\.mp4$", re.IGNORECASE)
FOLDER_RE = re.compile(r"^(.*?)\s*\((.*?)\)\s*$")

SPLITS = ["kp_train", "kp_val", "dev", "test"]
COLUMNS = ["video_id", "rel_path", "folder", "tube_label", "fill_level", "count", "count_bin",
           "fps", "n_frames", "duration_s", "width", "height", "orientation", "split"]

# Per-folder split sizes (each folder has 60 videos).
N_KP, N_KP_VAL, N_DEV = 12, 2, 8  # remainder -> test


def index_path() -> Path:
    return load_paths()["splits_dir"] / "video_index.csv"


def load_index(split: str | list[str] | None = None) -> pd.DataFrame:
    """Load the frozen index, optionally filtered to one or more splits."""
    df = pd.read_csv(index_path())
    if split is not None:
        df = df[df["split"].isin([split] if isinstance(split, str) else split)]
    return df.reset_index(drop=True)


def count_bin(count: int) -> str:
    if count <= 4:
        return "low"
    if count <= 8:
        return "mid"
    return "high"


def read_metadata(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"readable": False}
    # Phone recordings are usually variable-frame-rate: CAP_PROP_FPS is only an average,
    # so downstream timing should use per-frame timestamps (CAP_PROP_POS_MSEC).
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {
        "readable": True,
        "fps": round(fps, 3),
        "n_frames": n_frames,
        "duration_s": round(n_frames / fps, 2) if fps > 0 else np.nan,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return meta


def collect(video_root: Path) -> tuple[pd.DataFrame, list[str]]:
    """Index every `<id>_<count>.mp4` under video_root. Returns (index, skipped files)."""
    rows, skipped = [], []
    for folder in sorted(p for p in video_root.iterdir() if p.is_dir()):
        m = FOLDER_RE.match(folder.name)
        tube_label, fill_level = (m.group(1), m.group(2)) if m else (folder.name, "")
        for f in sorted(folder.iterdir()):
            fm = FILENAME_RE.match(f.name)
            if not fm:
                skipped.append(f.relative_to(video_root).as_posix())
                continue
            count = int(fm.group(2))
            rows.append({
                "video_id": int(fm.group(1)),
                "rel_path": f.relative_to(video_root).as_posix(),
                "folder": folder.name,
                "tube_label": tube_label,
                "fill_level": fill_level,
                "count": count,
                "count_bin": count_bin(count),
                **read_metadata(f),
            })
    return pd.DataFrame(rows), skipped


def sanity_checks(df: pd.DataFrame) -> list[str]:
    """Print a data summary and return a list of blocking errors (empty if OK)."""
    errors = []
    bad = df[~df["readable"]]
    if len(bad):
        errors.append(f"{len(bad)} unreadable videos:\n{bad[['rel_path']].to_string(index=False)}")
    bad_fps = df[df["readable"] & ~(df["fps"] > 0)]
    if len(bad_fps):
        errors.append(f"{len(bad_fps)} videos with fps <= 0:\n{bad_fps[['rel_path']].to_string(index=False)}")
    dups = df[df["video_id"].duplicated(keep=False)]
    if len(dups):
        errors.append(f"duplicate video_ids:\n{dups[['video_id', 'rel_path']].to_string(index=False)}")

    print("\nVideos per folder:\n" + df["folder"].value_counts().sort_index().to_string())
    print("\nFPS (average per video):\n" + df["fps"].describe().round(2).to_string())
    print("\nResolutions:\n" + (df["width"].astype(str) + "x" + df["height"].astype(str)).value_counts().to_string())
    print("\nDuration (s) by count -- should increase with count:")
    print(df.groupby("count")["duration_s"].agg(["count", "min", "median", "max"]).round(1).to_string())
    print(f"\nCorrelation(count, duration) = {df[['count', 'duration_s']].corr().iloc[0, 1]:.2f}")

    # Videos whose duration is far off the median for their count.
    med = df.groupby("count")["duration_s"].transform("median")
    outliers = df[(df["duration_s"] > 2 * med) | (df["duration_s"] < 0.5 * med)]
    if len(outliers):
        print(f"\nWARNING: {len(outliers)} duration outliers (>2x or <0.5x the median for their count) -- worth a look:")
        print(outliers[["rel_path", "count", "duration_s"]].to_string(index=False))
    return errors


def split_stratified(df: pd.DataFrame, n: int, seed: int):
    """Return (taken n rows, rest), stratified on count_bin when possible."""
    try:
        return train_test_split(df, train_size=n, stratify=df["count_bin"], random_state=seed)
    except ValueError:
        # Too few rows per bin for stratification (e.g. picking 2 of 12) -> plain random.
        return train_test_split(df, train_size=n, random_state=seed)


def assign_splits(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["split"] = ""
    for _, g in df.groupby("folder"):
        kp, rest = split_stratified(g, N_KP, seed)
        dev, test = split_stratified(rest, N_DEV, seed)
        kp_val, kp_train = split_stratified(kp, N_KP_VAL, seed)
        for part, name in [(kp_train, "kp_train"), (kp_val, "kp_val"), (dev, "dev"), (test, "test")]:
            df.loc[part.index, "split"] = name
    return df


def build_index(video_root: Path, seed: int = 42) -> pd.DataFrame:
    """Collect, check and split. Raises ValueError if the sanity checks find blocking errors."""
    df, skipped = collect(video_root)
    if skipped:
        print(f"Skipped {len(skipped)} non-video/unparseable files:")
        for s in skipped:
            print(f"  - {s}")
    print(f"\nIndexed {len(df)} videos from {video_root}")

    errors = sanity_checks(df)
    if errors:
        raise ValueError("Fix these before splitting:\n" + "\n".join(errors))

    df = assign_splits(df, seed)
    df["orientation"] = ""  # filled in later from keypoint length
    return df[COLUMNS].sort_values(["folder", "video_id"]).reset_index(drop=True)


def split_summary(df: pd.DataFrame) -> str:
    by_folder = pd.crosstab(df["folder"], df["split"], margins=True)[SPLITS + ["All"]]
    by_bin = pd.crosstab(df["count_bin"], df["split"], margins=True).reindex(["low", "mid", "high", "All"])[SPLITS + ["All"]]
    return f"Split x folder:\n{by_folder.to_string()}\n\nSplit x count_bin:\n{by_bin.to_string()}"

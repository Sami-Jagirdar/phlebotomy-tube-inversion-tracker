"""Stage 02: extract cropped frames for keypoint labelling -> data/frames/round<N>/.

Writes <video_id>_<frame_idx>.jpg crops plus manifest.csv
(video_id, split, frame_idx, t_ms, motion, kind, file, t_ms_check).

Usage:
    uv run scripts/02_extract_frames.py
    uv run scripts/02_extract_frames.py --n-motion 8 --min-gap-s 0.4
    uv run scripts/02_extract_frames.py --force    # wipe and redo this round (only before labelling!)
"""

import argparse
import sys
import time

import pandas as pd

from inversion_tracker.config import load_paths
from inversion_tracker.data.frames import extract_video
from inversion_tracker.data.index import load_index
from inversion_tracker.video import load_crop, video_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--splits", nargs="+", default=["kp_train", "kp_val"])
    ap.add_argument("--n-motion", type=int, default=6, help="high-motion frames per video")
    ap.add_argument("--n-calm", type=int, default=2, help="low-motion frames per video (start/end)")
    ap.add_argument("--min-gap-s", type=float, default=0.5, help="minimum time between picked frames")
    ap.add_argument("--force", action="store_true", help="delete this round's frames and redo it")
    args = ap.parse_args()

    out_dir = load_paths()["data_dir"] / "frames" / f"round{args.round}"
    manifest_path = out_dir / "manifest.csv"
    if manifest_path.exists() or any(out_dir.glob("*.jpg")):
        if not args.force:
            sys.exit(f"{out_dir} already has frames. Use --force to wipe and redo (never after labelling has started).")
        for f in out_dir.glob("*.jpg"):
            f.unlink()
        manifest_path.unlink(missing_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = load_crop()
    videos = load_index(args.splits)
    target = args.n_motion + args.n_calm
    print(f"Crop {crop}; {len(videos)} videos from {args.splits}; {target} frames/video -> {out_dir}\n")

    parts, t0 = [], time.time()
    for i, (_, row) in enumerate(videos.iterrows(), 1):
        part = extract_video(row, video_path(row.rel_path), crop, out_dir,
                             args.n_motion, args.n_calm, args.min_gap_s)
        parts.append(part)
        note = "" if len(part) == target else f"  <- only {len(part)} frames (short video)"
        print(f"[{i:3d}/{len(videos)}] video {row.video_id:4d} ({row.split}): {len(part)} frames{note}")

    manifest = pd.concat(parts, ignore_index=True)
    manifest.to_csv(manifest_path, index=False)

    missing = manifest["file"].isna().sum()
    drift = (manifest["t_ms"] - manifest["t_ms_check"]).abs().max()
    print(f"\nWrote {len(manifest)} frames + {manifest_path}  ({time.time() - t0:.0f}s)")
    print("\nFrames by split x kind:\n" + pd.crosstab(manifest["split"], manifest["kind"], margins=True).to_string())
    if missing:
        print(f"\nWARNING: {missing} selected frames were not written (decode ended early).")
    if drift > 1:
        print(f"\nWARNING: timestamps differ between passes by up to {drift:.1f} ms -- decoding is not deterministic.")


if __name__ == "__main__":
    main()

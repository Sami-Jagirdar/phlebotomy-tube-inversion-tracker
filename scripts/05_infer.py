"""Stage 05: run the pose model on every frame of a split's videos -> data/keypoints/<run>/.

Writes one <video_id>.parquet per video (frame_idx, t_ms, cap_x/y/conf, base_x/y/conf,
box_conf, box_x1/y1/x2/y2; crop pixels; NaN where nothing was detected), plus run_info.yaml
(the settings used) and summary.csv (per-video detection/confidence stats).
Everything after this stage reads these files and needs no GPU.

Videos that already have a parquet are skipped, so an interrupted run can simply be restarted.
<run> defaults to the training run's name (the folder above weights/).

Usage:
    uv run scripts/05_infer.py                                   # dev + test with r1_s_640
    uv run scripts/05_infer.py --splits test
    uv run scripts/05_infer.py --video-ids 181 182 --force
    uv run scripts/05_infer.py --weights outputs/runs/pose/r2_s_960/weights/best.pt --imgsz 960
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import ultralytics
import yaml
from ultralytics import YOLO

from inversion_tracker.config import REPO_ROOT, load_paths
from inversion_tracker.data_preprocessing.index import load_index
from inversion_tracker.inference import predict_video
from inversion_tracker.video import load_crop, video_path

LOW_CONF = 0.5  # summary only: fraction of frames with a keypoint confidence below this


def video_summary(df: pd.DataFrame) -> dict:
    detected = df["box_conf"].notna()
    return {
        "n_frames": len(df),
        "missed_frac": round(1 - detected.mean(), 4),
        "box_conf_median": df["box_conf"].median(),
        "cap_conf_median": df["cap_conf"].median(),
        "base_conf_median": df["base_conf"].median(),
        "cap_low_frac": round((df["cap_conf"].fillna(0) < LOW_CONF).mean(), 4),
        "base_low_frac": round((df["base_conf"].fillna(0) < LOW_CONF).mean(), 4),
    }


def check_run_info(info_path: Path, settings: dict, force: bool) -> None:
    """Refuse to mix parquets made with different settings in one folder."""
    if not info_path.exists() or force:
        return
    old = yaml.safe_load(info_path.read_text())
    changed = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
    if changed:
        sys.exit(f"{info_path.parent} was made with different settings {changed}.\n"
                 "Use --name for a new folder, or --force to redo every video.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["dev", "test"])
    ap.add_argument("--video-ids", nargs="+", type=int, help="only these videos (must be in --splits)")
    ap.add_argument("--weights", default="outputs/runs/pose/r1_s_640/weights/best.pt")
    ap.add_argument("--name", help="output folder under data/keypoints/ (default: the training run's name)")
    ap.add_argument("--imgsz", type=int, default=640, help="must match training")
    ap.add_argument("--conf", type=float, default=0.1,
                    help="box confidence threshold; kept low so the counter decides what to trust")
    ap.add_argument("--batch", type=int, default=32, help="frames per forward pass")
    ap.add_argument("--force", action="store_true", help="redo videos that already have a parquet")
    args = ap.parse_args()

    weights = (REPO_ROOT / args.weights).resolve()
    if not weights.exists():
        sys.exit(f"Weights not found: {weights}")
    name = args.name or weights.parent.parent.name
    out_dir = load_paths()["data_dir"] / "keypoints" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = load_crop()
    weights_str = weights.relative_to(REPO_ROOT).as_posix() if weights.is_relative_to(REPO_ROOT) else str(weights)
    settings = {"weights": weights_str, "imgsz": args.imgsz, "conf": args.conf,
                "crop": [crop.x, crop.y, crop.w, crop.h]}
    info_path = out_dir / "run_info.yaml"
    check_run_info(info_path, settings, args.force)
    info_path.write_text(yaml.safe_dump({**settings, "ultralytics": ultralytics.__version__,
                                         "updated": datetime.now().isoformat(timespec="seconds")}, sort_keys=False))

    videos = load_index(args.splits)
    if args.video_ids:
        videos = videos[videos["video_id"].isin(args.video_ids)]
    todo = videos if args.force else videos[[not (out_dir / f"{v}.parquet").exists() for v in videos["video_id"]]]
    print(f"{len(videos)} videos from {args.splits}; {len(videos) - len(todo)} already done; "
          f"{len(todo)} to run -> {out_dir}\n")

    model = YOLO(weights)
    total_frames, t0 = 0, time.time()
    for i, (_, row) in enumerate(todo.iterrows(), 1):
        t_start = time.time()
        df = predict_video(model, video_path(row.rel_path), crop, args.imgsz, args.conf, args.batch)
        tmp = out_dir / f"{row.video_id}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(out_dir / f"{row.video_id}.parquet")  # never leave a half-written parquet
        total_frames += len(df)
        s = video_summary(df)
        print(f"[{i:3d}/{len(todo)}] video {row.video_id:4d} ({row.split}): {len(df):4d} frames, "
              f"missed {s['missed_frac']:.1%}, cap<{LOW_CONF} {s['cap_low_frac']:.1%}, "
              f"base<{LOW_CONF} {s['base_low_frac']:.1%}  ({len(df) / (time.time() - t_start):.0f} fps)")
    if len(todo):
        print(f"\n{total_frames} frames in {time.time() - t0:.0f}s")

    # Summary over every video of the requested splits that has a parquet (including earlier runs).
    summary = pd.DataFrame([
        {"video_id": row.video_id, "split": row.split, "folder": row.folder, "count": row["count"],
         "duration_s": row.duration_s, **video_summary(pd.read_parquet(out_dir / f"{row.video_id}.parquet"))}
        for _, row in videos.iterrows() if (out_dir / f"{row.video_id}.parquet").exists()
    ])
    if summary.empty:
        sys.exit("No keypoint files for the requested videos.")
    summary_path = out_dir / "summary.csv"
    if summary_path.exists():  # keep rows for videos outside this call's splits/ids
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[~old["video_id"].isin(summary["video_id"])], summary], ignore_index=True)
    summary = summary.sort_values("video_id").reset_index(drop=True)
    summary.to_csv(summary_path, index=False)

    print(f"\nSummary ({len(summary)} videos) -> {summary_path}")
    print(summary.groupby("split")[["missed_frac", "cap_low_frac", "base_low_frac"]].mean().round(3).to_string())
    worst = summary.nlargest(5, "missed_frac")[["video_id", "split", "folder", "missed_frac", "cap_low_frac"]]
    print("\nMost missed frames:\n" + worst.to_string(index=False))


if __name__ == "__main__":
    main()

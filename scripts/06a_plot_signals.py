"""Stage 06a: plot the orientation signal s(t) per video -> outputs/reports/signals/<run>/.

For looking at the signal before building the counter. One PNG per video, named c<count>_<video_id>.png
so they sort by the file-name inversion count. The figure is described in counting/plots.py.

Usage:
    uv run scripts/06a_plot_signals.py                       # 12 dev videos spread over the counts
    uv run scripts/06a_plot_signals.py --all                 # every dev video
    uv run scripts/06a_plot_signals.py --video-ids 12 57 --T 0.3 --smooth-s 0.2
"""

import argparse

import numpy as np
import pandas as pd

from inversion_tracker.config import load_paths
from inversion_tracker.counting.plots import plot_signal
from inversion_tracker.counting.signal import SignalParams, compute_signal
from inversion_tracker.data_preprocessing.index import load_index


def pick_videos(videos: pd.DataFrame, n: int) -> pd.DataFrame:
    """n videos evenly spaced over the count-sorted list, so low, mid and high counts all appear."""
    videos = videos.sort_values(["count", "video_id"]).reset_index(drop=True)
    if n >= len(videos):
        return videos
    return videos.iloc[np.linspace(0, len(videos) - 1, n).round().astype(int)]


def main():
    defaults = SignalParams()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints", default="r1_s_640", help="folder under data/keypoints/")
    ap.add_argument("--splits", nargs="+", default=["dev"], help="tune on dev; keep test for the final evaluation")
    ap.add_argument("--video-ids", nargs="+", type=int)
    ap.add_argument("--n", type=int, default=12, help="how many videos (ignored with --all or --video-ids)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--T", type=float, default=0.4, help="hysteresis threshold drawn on the plot")
    ap.add_argument("--kpt-conf", type=float, default=defaults.kpt_conf)
    ap.add_argument("--max-gap-s", type=float, default=defaults.max_gap_s)
    ap.add_argument("--smooth-s", type=float, default=defaults.smooth_s)
    args = ap.parse_args()

    params = SignalParams(kpt_conf=args.kpt_conf, max_gap_s=args.max_gap_s, smooth_s=args.smooth_s)
    paths = load_paths()
    kp_dir = paths["data_dir"] / "keypoints" / args.keypoints
    out_dir = paths["outputs_dir"] / "reports" / "signals" / args.keypoints
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = load_index(args.splits)
    if args.video_ids:
        videos = videos[videos["video_id"].isin(args.video_ids)]
    elif not args.all:
        videos = pick_videos(videos, args.n)

    print(f"{len(videos)} videos, {params} -> {out_dir}\n")
    print(f"{'video':>5} {'count':>5} {'dur_s':>6} {'L_ref':>6} {'along_u':>8} {'dropped':>8}")
    for _, row in videos.iterrows():
        kp_path = kp_dir / f"{row.video_id}.parquet"
        if not kp_path.exists():
            print(f"{row.video_id:5d}  no keypoints file, skipped")
            continue
        try:
            sig = compute_signal(pd.read_parquet(kp_path), params)
        except ValueError as e:
            print(f"{row.video_id:5d}  skipped: {e}")
            continue
        plot_signal(row, sig, params.kpt_conf, args.T, out_dir / f"c{row['count']:02d}_{row.video_id}.png")
        print(f"{row.video_id:5d} {row['count']:5d} {row.duration_s:6.1f} {sig.l_ref:6.0f} "
              f"{sig.explained:8.0%} {1 - sig.frames['keep'].mean():8.0%}")


if __name__ == "__main__":
    main()

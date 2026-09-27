"""Stage 06a: plot the orientation signal s(t) per video -> outputs/reports/signals/<run>/.

For looking at the signal before building the counter. One PNG per video, named c<count>_<video_id>.png
so they sort by the file-name inversion count. Each figure shows:
    top left      s(t), raw (grey) and smoothed (blue), with the +-T hysteresis thresholds (red)
    middle left   |v| / L_ref -- dips mean the tube is pointing towards the camera
    bottom left   cap / base / box confidence
    right         every frame's v = cap - base as a point (colour = time), with the PCA axis u
Orange shading marks frames dropped from the signal (low confidence, not interpolated).

Usage:
    uv run scripts/06a_plot_signals.py                       # 12 dev videos spread over the counts
    uv run scripts/06a_plot_signals.py --all                 # every dev video
    uv run scripts/06a_plot_signals.py --video-ids 12 57 --T 0.3 --smooth-s 0.2
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inversion_tracker.config import load_paths
from inversion_tracker.counting.signal import Signal, SignalParams, compute_signal
from inversion_tracker.data_preprocessing.index import load_index

CAP_COLOR, BASE_COLOR, BOX_COLOR = "tab:red", "tab:blue", "tab:green"
DROP_COLOR = "tab:orange"


def pick_videos(videos: pd.DataFrame, n: int) -> pd.DataFrame:
    """n videos evenly spaced over the count-sorted list, so low, mid and high counts all appear."""
    videos = videos.sort_values(["count", "video_id"]).reset_index(drop=True)
    if n >= len(videos):
        return videos
    return videos.iloc[np.linspace(0, len(videos) - 1, n).round().astype(int)]


def shade_dropped(ax, t: np.ndarray, keep: np.ndarray) -> None:
    ax.fill_between(t, 0, 1, where=~keep, transform=ax.get_xaxis_transform(),
                    color=DROP_COLOR, alpha=0.2, linewidth=0, step="mid")


def plot_video(row: pd.Series, sig: Signal, params: SignalParams, T: float, out_path) -> None:
    f = sig.frames
    t, keep = f["t_s"].to_numpy(), f["keep"].to_numpy()

    fig = plt.figure(figsize=(16, 8), layout="constrained")
    gs = fig.add_gridspec(3, 2, width_ratios=[3, 1], height_ratios=[2, 1, 1])
    ax_s = fig.add_subplot(gs[0, 0])
    ax_l = fig.add_subplot(gs[1, 0], sharex=ax_s)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_s)
    ax_v = fig.add_subplot(gs[:, 1])

    ax_s.plot(t, f["s_raw"], color="0.7", lw=1, label="raw")
    ax_s.plot(t, f["s"], color="tab:blue", lw=1.6, label="smoothed")
    ax_s.axhline(0, color="0.5", lw=0.8)
    for y in (T, -T):
        ax_s.axhline(y, color="tab:red", ls="--", lw=1)
    ax_s.set_ylim(-1.4, 1.4)
    ax_s.set_ylabel("s(t)   (+ = start side)")
    ax_s.legend(loc="upper right", fontsize=8)
    ax_s.grid(alpha=0.3)

    ax_l.plot(t, f["length"] / sig.l_ref, color="k", lw=1)
    ax_l.set_ylim(0, 1.4)
    ax_l.set_ylabel("|v| / L_ref")
    ax_l.grid(alpha=0.3)

    ax_c.plot(t, f["cap_conf"], color=CAP_COLOR, lw=1, label="cap")
    ax_c.plot(t, f["base_conf"], color=BASE_COLOR, lw=1, label="base")
    ax_c.plot(t, f["box_conf"], color=BOX_COLOR, lw=1, label="box")
    ax_c.axhline(params.kpt_conf, color="0.4", ls=":", lw=1)
    ax_c.set_ylim(0, 1.05)
    ax_c.set_ylabel("confidence")
    ax_c.set_xlabel("time (s)")
    ax_c.legend(loc="lower right", fontsize=8, ncol=3)
    ax_c.grid(alpha=0.3)

    for ax in (ax_s, ax_l, ax_c):
        shade_dropped(ax, t, keep)

    sc = ax_v.scatter(f["vx"], f["vy"], c=t, cmap="viridis", s=6)
    a, b = sig.mean - sig.axis * sig.l_ref, sig.mean + sig.axis * sig.l_ref
    ax_v.plot([a[0], b[0]], [a[1], b[1]], color="tab:red", lw=1.5, label="PCA axis u")
    ax_v.annotate("", xy=tuple(b), xytext=tuple(sig.mean), arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.5})
    ax_v.plot(0, 0, "k+", ms=10)
    ax_v.set_aspect("equal", adjustable="datalim")
    ax_v.invert_yaxis()  # image coordinates: y points down
    ax_v.set_xlabel("vx (px)")
    ax_v.set_ylabel("vy (px, down)")
    ax_v.set_title(f"v = cap - base    along u: {sig.explained:.0%} of variance", fontsize=10)
    ax_v.legend(loc="upper right", fontsize=8)
    fig.colorbar(sc, ax=ax_v, label="time (s)", shrink=0.6)

    fig.suptitle(f"video {row.video_id}  |  {row.folder}  |  {row.split}  |  file count = {row['count']}  |  "
                 f"{row.duration_s:.1f} s  |  L_ref = {sig.l_ref:.0f} px  |  dropped {1 - keep.mean():.0%}",
                 fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


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
        plot_video(row, sig, params, args.T, out_dir / f"c{row['count']:02d}_{row.video_id}.png")
        print(f"{row.video_id:5d} {row['count']:5d} {row.duration_s:6.1f} {sig.l_ref:6.0f} "
              f"{sig.explained:8.0%} {1 - sig.frames['keep'].mean():8.0%}")


if __name__ == "__main__":
    main()

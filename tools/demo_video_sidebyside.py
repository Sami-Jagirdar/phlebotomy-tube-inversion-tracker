"""Same pipeline as tools/demo_video.py, but the s(t) plot is drawn into the video itself.

Writes one file to outputs/demo/<keypoints run>/: <video_id>_sidebyside.mp4, the annotated frame
(keypoints, arrow, running count) next to a s(t) plot that reveals itself as the video plays, so
the counting and the signal it comes from are visible side by side.

Uses configs/counter.yaml (the frozen signal + counter settings), same as tools/demo_video.py.

Usage:
    uv run tools/demo_video_sidebyside.py 195
    uv run tools/demo_video_sidebyside.py 195 --plot-width 700
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from inversion_tracker.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from inversion_tracker.counting.signal import SignalParams, compute_signal
from inversion_tracker.counting.state_machine import CounterParams, count_inversions
from inversion_tracker.data_preprocessing.index import load_index
from inversion_tracker.demo import SignalPanel, annotate_frame, running_count
from inversion_tracker.inference import predict_video
from inversion_tracker.video import iter_frames, load_crop, video_path


def load_counter(path):
    if not path.exists():
        raise SystemExit(f"{path} not found -- run scripts/06_tune_counter.py --write first.")
    cfg = load_yaml(path.name)
    return SignalParams(**cfg["signal"]), CounterParams(**cfg["counter"]), cfg["keypoints"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video_id", type=int)
    p.add_argument("--weights", default="outputs/runs/pose/r1_s_640/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640, help="must match training")
    p.add_argument("--conf", type=float, default=0.1, help="box confidence threshold")
    p.add_argument("--batch", type=int, default=32, help="frames per forward pass")
    p.add_argument("--plot-width", type=int, default=640, help="pixel width of the plot panel")
    args = p.parse_args()

    sp, cp, run_name = load_counter(CONFIG_DIR / "counter.yaml")

    rows = load_index()
    rows = rows[rows["video_id"] == args.video_id]
    if rows.empty:
        raise SystemExit(f"video_id {args.video_id} not in the index")
    row = rows.iloc[0]

    crop = load_crop()
    model = YOLO(REPO_ROOT / args.weights)

    print(f"video {row.video_id} ({row.split}, {row['count']} inversions): {row.rel_path}")
    path = video_path(row.rel_path)
    kp = predict_video(model, path, crop, args.imgsz, args.conf, args.batch)
    sig = compute_signal(kp, sp)
    f = sig.frames
    t, s = f["t_s"].to_numpy(), f["s"].to_numpy()
    res = count_inversions(t, s, cp)
    counts = running_count(t, res.kept)

    out_dir = load_paths()["outputs_dir"] / "demo" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{row.video_id}_sidebyside.mp4"

    panel = SignalPanel(t, s, cp.T, res.flips, set(res.kept.tolist()), crop.h, args.plot_width)
    # VFR source: the writer uses the average fps, so playback speed is only approximate.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(row.fps),
                              (crop.w + args.plot_width, crop.h))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open video writer for {out_path}")
    try:
        for i, (_, _, frame) in enumerate(iter_frames(path, crop)):
            frame = annotate_frame(frame, kp.iloc[i], sp.kpt_conf, bool(f["keep"].iloc[i]),
                                    float(s[i]), cp.T, counts[i])
            plot_img = panel.render(i, counts[i])
            if plot_img.shape[:2] != (crop.h, args.plot_width):  # matplotlib rounds figsize*dpi to whole pixels
                plot_img = cv2.resize(plot_img, (args.plot_width, crop.h))
            writer.write(np.hstack([frame, plot_img]))
    finally:
        writer.release()
        panel.close()
    print(f"-> {out_path}  (predicted {res.count:g}, file-name count {row['count']})")


if __name__ == "__main__":
    main()

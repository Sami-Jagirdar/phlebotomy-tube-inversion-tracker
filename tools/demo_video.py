"""Run the frozen pipeline (pose model -> signal -> counter) on one video for the demo.

Writes two files to outputs/demo/<keypoints run>/:
    <video_id>_video.mp4    the video with keypoints, arrow, dropped-frame tint, s(t) readout
                            and a running inversion count
    <video_id>_signal.png   the diagnostic signal plot (counting/plots.py) with this run's flips

Uses configs/counter.yaml (the frozen signal + counter settings) so the demo matches the reported
numbers. See tools/demo_video_sidebyside.py for a version with the plot drawn into the video itself.

Usage:
    uv run tools/demo_video.py 195
    uv run tools/demo_video.py 195 --weights outputs/runs/pose/r1_s_640/weights/best.pt --imgsz 640
"""

import argparse

import cv2
from ultralytics import YOLO

from inversion_tracker.config import CONFIG_DIR, REPO_ROOT, load_paths, load_yaml
from inversion_tracker.counting.plots import plot_signal
from inversion_tracker.counting.signal import SignalParams, compute_signal
from inversion_tracker.counting.state_machine import CounterParams, count_inversions
from inversion_tracker.data_preprocessing.index import load_index
from inversion_tracker.demo import annotate_frame, running_count
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
    video_out = out_dir / f"{row.video_id}_video.mp4"
    plot_out = out_dir / f"{row.video_id}_signal.png"

    # VFR source: the writer uses the average fps, so playback speed is only approximate.
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), float(row.fps), (crop.w, crop.h))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open video writer for {video_out}")
    try:
        for i, (_, _, frame) in enumerate(iter_frames(path, crop)):
            frame = annotate_frame(frame, kp.iloc[i], sp.kpt_conf, bool(f["keep"].iloc[i]),
                                    float(s[i]), cp.T, counts[i])
            writer.write(frame)
    finally:
        writer.release()
    print(f"-> {video_out}")

    plot_signal(row, sig, sp.kpt_conf, cp.T, plot_out, res)
    print(f"-> {plot_out}  (predicted {res.count:g}, file-name count {row['count']})")


if __name__ == "__main__":
    main()

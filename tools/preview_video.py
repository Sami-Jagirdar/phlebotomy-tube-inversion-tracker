"""Run the pose model on one video and save an annotated preview -> outputs/previews/.

Frames are cropped exactly as in training. Drawn per frame: the best detection's box (green),
cap (red) and base (blue) with a base->cap arrow, plus frame index, time and confidences.
A keypoint with confidence below --kpt-conf is drawn as a ring instead of a filled dot.

Usage:
    uv run tools/preview_video.py                     # random kp_val video
    uv run tools/preview_video.py 64
    uv run tools/preview_video.py 64 --weights outputs/runs/pose/r1_s_640/weights/best.pt --imgsz 640
"""

import argparse

import cv2
import numpy as np
from ultralytics import YOLO

from inversion_tracker.config import REPO_ROOT, load_paths
from inversion_tracker.data_preprocessing.index import load_index
from inversion_tracker.video import iter_frames, load_crop, video_path

BOX_COLOR = (0, 255, 0)       # green
CAP_COLOR = (0, 0, 255)       # red
BASE_COLOR = (255, 0, 0)      # blue
ARROW_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 255, 255)    # yellow


def pick_video(video_id: int | None, seed: int | None):
    df = load_index()
    if video_id is None:
        return df[df["split"] == "kp_val"].sample(1, random_state=seed).iloc[0]
    rows = df[df["video_id"] == video_id]
    if rows.empty:
        raise SystemExit(f"video_id {video_id} not in the index")
    row = rows.iloc[0]
    return row


def draw_point(frame: np.ndarray, pt: tuple[int, int], color, visible: bool) -> None:
    cv2.circle(frame, pt, 9, color, -1 if visible else 3, cv2.LINE_AA)


def annotate(frame: np.ndarray, result, kpt_conf: float) -> str:
    """Draw the highest-confidence detection onto frame; return a status string."""
    if not len(result.boxes):
        return "no detection"
    i = int(result.boxes.conf.argmax())  # exactly one tube per frame
    box_conf = float(result.boxes.conf[i])
    x1, y1, x2, y2 = result.boxes.xyxy[i].cpu().numpy().round().astype(int)
    (cx, cy), (bx, by) = result.keypoints.xy[i].cpu().numpy().round().astype(int)
    confs = result.keypoints.conf
    c_conf, b_conf = confs[i].cpu().numpy() if confs is not None else (1.0, 1.0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
    cv2.arrowedLine(frame, (bx, by), (cx, cy), ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.08)
    draw_point(frame, (cx, cy), CAP_COLOR, c_conf >= kpt_conf)
    draw_point(frame, (bx, by), BASE_COLOR, b_conf >= kpt_conf)
    return f"box {box_conf:.2f}  cap {c_conf:.2f}  base {b_conf:.2f}"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video_id", type=int, nargs="?", help="default: a random kp_val video")
    p.add_argument("--weights", default="outputs/runs/pose/r1_s_640/weights/best.pt")
    p.add_argument("--imgsz", type=int, default=640, help="must match training")
    p.add_argument("--conf", type=float, default=0.25, help="box confidence threshold")
    p.add_argument("--kpt-conf", type=float, default=0.5, help="below this a keypoint is drawn as a ring")
    p.add_argument("--seed", type=int, help="seed for the random video pick")
    args = p.parse_args()

    row = pick_video(args.video_id, args.seed)
    crop = load_crop()
    model = YOLO(REPO_ROOT / args.weights)

    out_dir = load_paths()["outputs_dir"] / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{row['video_id']}_{row['split']}_{row['count']}inv.mp4"
    # VFR source: the writer uses the average fps, so playback speed is only approximate.
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(row["fps"]), (crop.w, crop.h))
    if not writer.isOpened():
        raise SystemExit(f"Cannot open video writer for {out_path}")

    print(f"video {row['video_id']} ({row['split']}, {row['count']} inversions): {row['rel_path']}")
    n_frames = n_missed = 0
    try:
        for idx, t_ms, frame in iter_frames(video_path(row["rel_path"]), crop):
            frame = np.ascontiguousarray(frame)  # the crop is a view; cv2 drawing needs contiguous memory
            result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
            status = annotate(frame, result, args.kpt_conf)
            n_missed += status == "no detection"
            cv2.putText(frame, f"#{idx}  {t_ms / 1000:6.2f}s  {status}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, TEXT_COLOR, 2, cv2.LINE_AA)
            writer.write(frame)
            n_frames += 1
    finally:
        writer.release()

    print(f"{n_frames} frames, {n_missed} without a detection -> {out_path}")


if __name__ == "__main__":
    main()

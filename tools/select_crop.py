"""Interactively choose the fixed crop box -> configs/crop.yaml.

Shows ~10 videos spread across the video_id range (so every recording session and folder
is covered). For each one, scrub to frames where the tube is furthest out (mid-inversion)
and draw a box around the region the tube reaches. The final crop is the union of all boxes
plus a margin, made square where possible and clipped to the frame.

Controls (in the video window):
    trackbar / a,d    scrub (a/d = -/+3 frames)
    r                 draw a box on the current frame (drag, then ENTER/SPACE; c cancels)
    n                 next video
    q                 finish early

Usage:
    uv run tools/select_crop.py
    uv run tools/select_crop.py --n-videos 15 --margin 0.15
"""

import argparse

import cv2
import numpy as np

from inversion_tracker.data_preprocessing.index import load_index
from inversion_tracker.video import CROP_FILE, Crop, save_crop, video_path

WIN = "select crop"
UNION_COLOR = (0, 255, 255)  # yellow: union so far
BOX_COLOR = (0, 200, 0)      # green: boxes drawn on this video


def pick_videos(n: int):
    """Evenly spaced across video_id order. Test videos are excluded to keep them untouched."""
    df = load_index(["kp_train", "kp_val", "dev"]).sort_values("video_id").reset_index(drop=True)
    return df.iloc[np.linspace(0, len(df) - 1, n).round().astype(int)]


def union(boxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return x0, y0, x1 - x0, y1 - y0


def finalize(box: tuple[int, int, int, int], margin: float, frame_w: int, frame_h: int) -> Crop:
    """Add margin, expand to square (as far as the frame allows), shift/clip into the frame."""
    x, y, w, h = box
    cx, cy = x + w / 2, y + h / 2
    side_w = side_h = max(w, h) * (1 + 2 * margin)
    side_w, side_h = min(side_w, frame_w), min(side_h, frame_h)
    x0 = int(round(min(max(cx - side_w / 2, 0), frame_w - side_w)))
    y0 = int(round(min(max(cy - side_h / 2, 0), frame_h - side_h)))
    w_out, h_out = int(side_w) // 2 * 2, int(side_h) // 2 * 2  # even sizes play nicer with video codecs
    return Crop(x0, y0, w_out, h_out)


def draw(frame, boxes, current_union, scale, label):
    vis = frame.copy()
    for b in boxes:
        cv2.rectangle(vis, (b[0], b[1]), (b[0] + b[2], b[1] + b[3]), BOX_COLOR, 3)
    if current_union:
        u = current_union
        cv2.rectangle(vis, (u[0], u[1]), (u[0] + u[2], u[1] + u[3]), UNION_COLOR, 3)
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.putText(vis, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return vis


def select_on_video(row, all_boxes, scale) -> bool:
    """Returns False if the user pressed q."""
    cap = cv2.VideoCapture(str(video_path(row.rel_path)))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cv2.namedWindow(WIN)
    cv2.createTrackbar("frame", WIN, n_frames // 2, max(n_frames - 1, 1), lambda _: None)

    video_boxes, shown_idx, frame = [], -1, None
    label = f"video {row.video_id} ({row.folder}, n={row['count']})  r=box  n=next  q=finish"
    try:
        while True:
            idx = cv2.getTrackbarPos("frame", WIN)
            if idx != shown_idx:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, f = cap.read()
                if ok:
                    frame, shown_idx = f, idx
            if frame is not None:
                u = union(all_boxes) if all_boxes else None
                cv2.imshow(WIN, draw(frame, video_boxes, u, scale, label))

            key = cv2.waitKey(30) & 0xFF
            if key == ord("a"):
                cv2.setTrackbarPos("frame", WIN, max(idx - 3, 0))
            elif key == ord("d"):
                cv2.setTrackbarPos("frame", WIN, min(idx + 3, n_frames - 1))
            elif key == ord("r") and frame is not None:
                small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                x, y, w, h = cv2.selectROI("draw box", small, showCrosshair=False)
                cv2.destroyWindow("draw box")
                if w > 0 and h > 0:
                    box = tuple(int(round(v / scale)) for v in (x, y, w, h))
                    video_boxes.append(box)
                    all_boxes.append(box)
            elif key == ord("n"):
                return True
            elif key == ord("q"):
                return False
    finally:
        cap.release()
        cv2.destroyWindow(WIN)  # fresh window (and trackbar range) for the next video


def preview(crop: Crop, videos, scale) -> bool:
    """Show the final crop on the middle frame of each video. y = save, anything else = discard."""
    for _, row in videos.iterrows():
        cap = cv2.VideoCapture(str(video_path(row.rel_path)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) // 2)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        cv2.rectangle(frame, (crop.x, crop.y), (crop.x + crop.w, crop.y + crop.h), UNION_COLOR, 3)
        vis = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(vis, f"final crop {crop.w}x{crop.h}   y=save  any other key=next/discard", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(WIN, vis)
        if (cv2.waitKey(0) & 0xFF) == ord("y"):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-videos", type=int, default=10)
    ap.add_argument("--margin", type=float, default=0.12, help="fraction of the box size added on each side")
    ap.add_argument("--scale", type=float, default=0.6, help="display scale (1920x1080 -> 1152x648 at 0.6)")
    args = ap.parse_args()

    if CROP_FILE.exists():
        print(f"Note: {CROP_FILE} exists and will be overwritten if you save. "
              "Labels made with the old crop would no longer line up.")

    videos = pick_videos(args.n_videos)
    boxes: list[tuple[int, int, int, int]] = []
    for _, row in videos.iterrows():
        if not select_on_video(row, boxes, args.scale):
            break

    if not boxes:
        cv2.destroyAllWindows()
        raise SystemExit("No boxes drawn -- nothing saved.")

    frame_w, frame_h = int(videos.iloc[0].width), int(videos.iloc[0].height)
    crop = finalize(union(boxes), args.margin, frame_w, frame_h)
    print(f"Union of {len(boxes)} boxes: {union(boxes)}  ->  final crop: {crop}")

    save = preview(crop, videos, args.scale)
    cv2.destroyAllWindows()
    if save:
        save_crop(crop, (frame_w, frame_h))
        print(f"Saved {CROP_FILE}")
    else:
        print("Discarded -- nothing saved.")


if __name__ == "__main__":
    main()

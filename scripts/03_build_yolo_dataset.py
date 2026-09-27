"""Stage 03: CVAT export -> training-ready YOLO pose dataset -> data/yolo_dataset/.

Rebuilds the boxes (padded around the keypoints), zeroes outside keypoints, splits images into
train/val by their video's split (kp_train -> train, kp_val -> val; never by frame), writes
data/yolo_dataset/dataset.yaml, and saves overlay images for a visual check.

Usage:
    uv run scripts/03_build_yolo_dataset.py
    uv run scripts/03_build_yolo_dataset.py --force        # rebuild from scratch
"""

import argparse
import shutil
import sys

import pandas as pd

from inversion_tracker.config import load_paths
from inversion_tracker.data_preprocessing.yolo import build_dataset, save_check_images, write_dataset_yaml


def main():
    paths = load_paths()
    data = paths["data_dir"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=lambda p: data / p, default=data / "images" / "train",
                    help="labelled frames + manifest.csv (relative to data/)")
    ap.add_argument("--labels", type=lambda p: data / p, default=data / "round1_labels" / "labels" / "train",
                    help="CVAT label .txt files (relative to data/)")
    ap.add_argument("--out", type=lambda p: data / p, default=data / "yolo_dataset", help="relative to data/")
    ap.add_argument("--n-check", type=int, default=20, help="random overlay images (plus edge cases)")
    ap.add_argument("--force", action="store_true", help="delete and rebuild the output dataset")
    args = ap.parse_args()

    if args.out.exists():
        if not args.force:
            sys.exit(f"{args.out} already exists. Use --force to rebuild it.")
        shutil.rmtree(args.out)

    manifest = pd.read_csv(args.images / "manifest.csv")
    report = build_dataset(args.images, args.labels, manifest, args.out)
    yaml_path = write_dataset_yaml(args.out)
    report.to_csv(args.out / "build_report.csv", index=False)

    ok = report[report["status"] == "ok"]
    print(f"Images found: {len(report)}   written: {len(ok)}")
    print("\nStatus:\n" + report["status"].value_counts().to_string())
    print("\nImages per split:\n" + ok["split"].value_counts().to_string())
    print(f"\nVideos per split: {ok.assign(vid=ok.file.str.split('_').str[0]).groupby('split').vid.nunique().to_dict()}")
    print("\nVisibility (cap, base):\n" + ok.groupby(["v_cap", "v_base"]).size().to_string())
    print(f"\nSingle-point frames (box = +-75 px around the one point): {(ok['n_labelled'] == 1).sum()}")

    sizes = pd.DataFrame({
        "orig_w": ok["orig_w_px"], "orig_h": ok["orig_h_px"], "new_w": ok["new_w_px"], "new_h": ok["new_h_px"]})
    print("\nBox size in px, before -> after (expect new minimum around 90):\n" + sizes.describe().round(1).loc[
        ["min", "50%", "max"]].to_string())

    problems = report[report["status"] != "ok"]
    if len(problems):
        print("\nWARNING: images not written:\n" + problems[["file", "split", "status"]].to_string(index=False))

    check_dir = paths["outputs_dir"] / "reports" / "label_check"
    if check_dir.exists():
        shutil.rmtree(check_dir)
    picks = save_check_images(args.out, report, check_dir, args.n_check)
    print(f"\nWrote {yaml_path}\nOverlay check: {len(picks)} images in {check_dir}"
          "\n  green = box, red = cap, blue = base; filled = visible, ring = occluded")


if __name__ == "__main__":
    main()

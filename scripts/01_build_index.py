"""Stage 01: build the video index and freeze the split -> splits/video_index.csv.

Usage:
    uv run scripts/01_build_index.py
    uv run scripts/01_build_index.py --force   # re-draw the split (only before any results exist!)
"""

import argparse
import sys

from inversion_tracker.config import load_paths
from inversion_tracker.data.index import build_index, index_path, split_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="overwrite an existing index (re-draws the split!)")
    args = ap.parse_args()

    out = index_path()
    if out.exists() and not args.force:
        sys.exit(f"{out} already exists -- the split is frozen. Use --force only if you really mean to re-draw it.")

    try:
        df = build_index(load_paths()["video_root"], seed=args.seed)
    except ValueError as e:
        sys.exit(f"\n{e}")

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWrote {out}\n\n{split_summary(df)}")


if __name__ == "__main__":
    main()

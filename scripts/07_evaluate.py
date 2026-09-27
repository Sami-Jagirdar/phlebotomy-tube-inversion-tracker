"""Stage 07: evaluate the frozen counter (configs/counter.yaml) -> outputs/reports/eval/<keypoints>/<split>/.

Run on test ONCE, after configs/counter.yaml is committed. The script refuses to overwrite an
existing evaluation unless --force is given, so a second test run is a deliberate choice.
Running it on dev first is a good check: it should reproduce the dev_score stored in counter.yaml.

Writes:
    per_video.csv       true vs. predicted count, error, flips (half_count = counted flips / 2, unrounded),
                        seconds per inversion, signal stats
    summary.csv         scores overall and by folder / count bin / true count
                        (exact; within05 = unrounded half_count within 0.5 of truth, 0.5 included;
                        within1; mae; bias)
    confusion.csv/.png  true count vs. predicted count (predictions can be halves, e.g. 3.5)
    errors.png          distribution of predicted - true
    speed_by_folder.csv seconds per inversion by tube group
    counter_used.yaml   copy of the settings that produced these numbers
    misses/             signal plots of every video that isn't counted exactly, with the flips marked

Usage:
    uv run scripts/07_evaluate.py --splits dev       # sanity check: should match counter.yaml's dev_score
    uv run scripts/07_evaluate.py                    # the one test run
"""

import argparse
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from inversion_tracker.config import CONFIG_DIR, load_paths, load_yaml
from inversion_tracker.counting.metrics import score
from inversion_tracker.counting.plots import plot_signal
from inversion_tracker.counting.signal import SignalParams, compute_signal
from inversion_tracker.counting.state_machine import CounterParams, count_inversions
from inversion_tracker.data_preprocessing.index import load_index

COUNTER_FILE = CONFIG_DIR / "counter.yaml"
INK, MUTED, GRID = "#1f1f1f", "#6b6b6b", "#e6e6e6"
BAR_COLOR = "#3a6ea5"


def evaluate_video(row: pd.Series, kp: pd.DataFrame, sp: SignalParams, cp: CounterParams):
    """Returns (per-video record, Signal or None, CountResult or None)."""
    rec = {"video_id": row.video_id, "split": row.split, "folder": row.folder, "count_bin": row.count_bin,
           "count": row["count"], "duration_s": row.duration_s}
    try:
        sig = compute_signal(kp, sp)
    except ValueError as e:  # too few confident frames: nothing to count
        return {**rec, "pred": 0.0, "half_count": 0.0, "note": f"no signal: {e}"}, None, None
    f = sig.frames
    res = count_inversions(f["t_s"].tolist(), f["s"].tolist(), cp)
    kept = res.kept
    # 2 flips per inversion, so seconds per inversion = 2 x the mean gap between counted flips.
    s_per_inv = 2 * (kept[-1] - kept[0]) / (len(kept) - 1) if len(kept) >= 2 else np.nan
    return {**rec, "pred": res.count, "half_count": len(kept) / 2, "n_flips": len(res.flips), "n_kept": len(kept),
            "s_per_inversion": s_per_inv, "l_ref_px": sig.l_ref, "along_u": sig.explained,
            "dropped_frac": 1 - f["keep"].mean(), "flip_times": ";".join(f"{x:.2f}" for x in res.flips),
            "note": ""}, sig, res


def summarise(per_video: pd.DataFrame) -> pd.DataFrame:
    def s(g: pd.DataFrame) -> dict:
        return score(g["pred"], g["count"], g["half_count"])

    rows = [{"group_by": "all", "group": "all", "n": len(per_video), **s(per_video)}]
    for col in ["folder", "count_bin", "count"]:
        for key, g in per_video.groupby(col, sort=True):
            rows.append({"group_by": col, "group": key, "n": len(g), **s(g)})
    return pd.DataFrame(rows)


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def plot_confusion(ct: pd.DataFrame, title: str, out_path) -> None:
    fig, ax = plt.subplots(figsize=(max(6.0, 0.42 * len(ct.columns) + 2.5), 6), layout="constrained")
    im = ax.imshow(ct.to_numpy(), cmap="Blues", aspect="auto")
    vals, vmax = ct.to_numpy(), max(ct.to_numpy().max(), 1)
    for i, j in np.argwhere(vals > 0):
        ax.text(j, i, vals[i, j], ha="center", va="center", fontsize=8,
                color="white" if vals[i, j] > 0.6 * vmax else INK)
    cols = list(ct.columns)
    for i, true_count in enumerate(ct.index):  # outline the exact-match cells
        if true_count in cols:
            j = cols.index(true_count)
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1))
    ax.set_xticks(range(len(cols)), [f"{c:g}" for c in cols])
    ax.set_yticks(range(len(ct.index)), ct.index)
    ax.set_xlabel("predicted inversions", color=INK)
    ax.set_ylabel("file-name count", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    style(ax)
    fig.colorbar(im, ax=ax, label="videos", shrink=0.7)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_errors(err: np.ndarray, title: str, out_path) -> None:
    lo, hi = min(err.min(), -1.0), max(err.max(), 1.0)
    x = np.arange(lo, hi + 0.25, 0.5)
    heights = np.array([(err == v).sum() for v in x])
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    bars = ax.bar(x, heights, width=0.38, color=BAR_COLOR)
    ax.bar_label(bars, labels=[str(h) if h else "" for h in heights], color=INK, fontsize=9, padding=2)
    ax.set_xticks(x, ["0" if v == 0 else f"{v:+g}" for v in x])
    ax.set_xlabel("predicted - file-name count (inversions)", color=INK)
    ax.set_ylabel("videos", color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    style(ax)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--plots", choices=["misses", "all", "none"], default="misses", help="which signal plots to save")
    ap.add_argument("--force", action="store_true", help="overwrite an existing evaluation of these splits")
    args = ap.parse_args()

    if not COUNTER_FILE.exists():
        sys.exit(f"{COUNTER_FILE} not found -- run scripts/06_tune_counter.py --write first.")
    cfg = load_yaml(COUNTER_FILE.name)
    sp, cp = SignalParams(**cfg["signal"]), CounterParams(**cfg["counter"])

    paths = load_paths()
    kp_dir = paths["data_dir"] / "keypoints" / cfg["keypoints"]
    split_name = "+".join(args.splits)
    out_dir = paths["outputs_dir"] / "reports" / "eval" / cfg["keypoints"] / split_name
    if (out_dir / "per_video.csv").exists() and not args.force:
        sys.exit(f"{out_dir} already has an evaluation. The test set is meant to be run once; "
                 "use --force only if you really mean to redo it.")
    videos = load_index(args.splits)
    missing = [v for v in videos["video_id"] if not (kp_dir / f"{v}.parquet").exists()]
    if missing:
        sys.exit(f"No keypoints in {kp_dir} for {len(missing)} videos (e.g. {missing[:5]}) -- run scripts/05_infer.py.")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "misses").mkdir(parents=True)
    shutil.copy(COUNTER_FILE, out_dir / "counter_used.yaml")

    print(f"{len(videos)} videos from {args.splits}\nsignal:  {sp}\ncounter: {cp}\n-> {out_dir}\n")
    records = []
    for _, row in videos.iterrows():
        rec, sig, res = evaluate_video(row, pd.read_parquet(kp_dir / f"{row.video_id}.parquet"), sp, cp)
        records.append(rec)
        err = rec["pred"] - row["count"]
        if sig is not None and (args.plots == "all" or (args.plots == "misses" and err != 0)):
            name = f"err{err:+g}_c{row['count']:02d}_{row.video_id}.png"
            plot_signal(row, sig, sp.kpt_conf, cp.T, out_dir / "misses" / name, res)

    per_video = pd.DataFrame(records)
    per_video["err"] = per_video["pred"] - per_video["count"]
    per_video = per_video.sort_values("video_id").reset_index(drop=True)
    per_video.to_csv(out_dir / "per_video.csv", index=False)

    summary = summarise(per_video)
    summary.to_csv(out_dir / "summary.csv", index=False)

    ct = pd.crosstab(per_video["count"], per_video["pred"]).reindex(
        range(int(per_video["count"].min()), int(per_video["count"].max()) + 1), fill_value=0)
    ct.to_csv(out_dir / "confusion.csv")
    n = len(per_video)
    plot_confusion(ct, f"Predicted vs. file-name count ({split_name}, n={n})", out_dir / "confusion.png")
    plot_errors(per_video["err"].to_numpy(), f"Counting error ({split_name}, n={n})", out_dir / "errors.png")

    speed = per_video.groupby("folder")["s_per_inversion"].agg(
        n="count", median="median", p25=lambda x: x.quantile(0.25), p75=lambda x: x.quantile(0.75))
    speed.round(2).to_csv(out_dir / "speed_by_folder.csv")

    fmt = {c: "{:.1%}".format for c in ["exact", "within05", "within1"]} | {"mae": "{:.2f}".format,
                                                                             "bias": "{:+.2f}".format}
    overall = summary.iloc[0]
    print(f"Overall: exact {overall.exact:.1%}, within 0.5 {overall.within05:.1%}, within 1 {overall.within1:.1%}, "
          f"MAE {overall.mae:.2f}, bias {overall.bias:+.2f}")
    if "dev_score" in cfg and args.splits == ["dev"]:
        print(f"(counter.yaml dev_score: {cfg['dev_score']})")
    for col in ["folder", "count_bin"]:
        print(f"\nBy {col}:\n" + summary[summary["group_by"] == col].drop(columns="group_by")
              .to_string(index=False, formatters=fmt))
    print("\nErrors (predicted - true): " + ", ".join(
        f"{v:+g}: {c}" for v, c in per_video["err"].value_counts().sort_index().items()))
    no_signal = (per_video["note"] != "").sum()
    if no_signal:
        print(f"WARNING: {no_signal} videos had no usable signal and were counted as 0 (see note column).")
    print(f"\nSeconds per inversion by folder:\n{speed.round(2).to_string()}")
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()

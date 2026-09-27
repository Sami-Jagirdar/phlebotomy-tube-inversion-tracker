"""Stage 06: grid-search the signal + counter settings on dev -> outputs/reports/counter_tuning/<run>/.

Every combination in configs/counter_grid.yaml is scored against the file-name counts:
    exact    fraction of videos counted exactly right (the main score)
    within05 fraction whose unrounded count (counted flips / 2) is within 0.5 of the truth, 0.5 included
             (e.g. truth 2 and 3 flips = 1.5 counts, even though floor predicts 1)
    within1  fraction off by at most one inversion
    mae      mean absolute error, in inversions (counts can be halves, e.g. 1.5)
    bias     mean (predicted - true); negative = undercounting
    robust   mean exact over the combination and its grid neighbours (one step in one setting);
             used to break ties, so a flat region beats a lucky single point

Writes grid_results.csv (all combinations, best first) and per_video.csv (best vs. the untuned
baseline: default signal settings, T=0.5, no dwell, no edge dropping, floor). With --write, the best
settings are frozen to configs/counter.yaml; commit that file before running anything on test.

Test is not allowed here: tuning on it would make the final numbers look better than they are.

Usage:
    uv run scripts/06_tune_counter.py
    uv run scripts/06_tune_counter.py --top 30
    uv run scripts/06_tune_counter.py --write        # freeze the winner to configs/counter.yaml
"""

import argparse
import time
from dataclasses import asdict
from datetime import date
from itertools import product

import numpy as np
import pandas as pd
import yaml

from inversion_tracker.config import CONFIG_DIR, load_paths, load_yaml
from inversion_tracker.counting.metrics import score
from inversion_tracker.counting.signal import SignalParams, compute_signal
from inversion_tracker.counting.state_machine import CounterParams, count_from_flips, count_inversions, detect_flips
from inversion_tracker.data_preprocessing.index import load_index

COUNTER_FILE = CONFIG_DIR / "counter.yaml"
BASELINE = (SignalParams(), CounterParams())


def signals_for(kps: list[pd.DataFrame], sp: SignalParams) -> list:
    """(t, s) per video, or None where the video has too few confident frames (counted as 0)."""
    out = []
    for kp in kps:
        try:
            f = compute_signal(kp, sp).frames
            out.append((f["t_s"].tolist(), f["s"].tolist()))  # plain lists: much faster to loop over
        except ValueError:
            out.append(None)
    return out


def run(kps: list[pd.DataFrame], sp: SignalParams, cp: CounterParams) -> list:
    return [count_inversions(*sig, cp) if sig else None for sig in signals_for(kps, sp)]


def neighbour_mean(keys: list[tuple], exact: np.ndarray) -> np.ndarray:
    """Mean exact over each combination and its neighbours (one index step in one setting)."""
    lookup = {k: e for k, e in zip(keys, exact)}
    out = np.empty(len(keys))
    for i, k in enumerate(keys):
        vals = [exact[i]]
        for d in range(len(k)):
            for step in (-1, 1):
                nb = k[:d] + (k[d] + step,) + k[d + 1:]
                if nb in lookup:
                    vals.append(lookup[nb])
        out[i] = np.mean(vals)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keypoints", default="r1_s_640", help="folder under data/keypoints/")
    ap.add_argument("--splits", nargs="+", default=["dev"], choices=["dev", "kp_train", "kp_val"])
    ap.add_argument("--grid", default="counter_grid.yaml", help="file in configs/")
    ap.add_argument("--top", type=int, default=15, help="how many combinations to print")
    ap.add_argument("--write", action="store_true", help=f"freeze the best settings to {COUNTER_FILE.name}")
    args = ap.parse_args()

    paths = load_paths()
    kp_dir = paths["data_dir"] / "keypoints" / args.keypoints
    out_dir = paths["outputs_dir"] / "reports" / "counter_tuning" / args.keypoints
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = load_index(args.splits)
    have = videos["video_id"].map(lambda v: (kp_dir / f"{v}.parquet").exists())
    if not have.all():
        print(f"WARNING: no keypoints for videos {videos.loc[~have, 'video_id'].tolist()} -- left out")
    videos = videos[have].reset_index(drop=True)
    kps = [pd.read_parquet(kp_dir / f"{v}.parquet") for v in videos["video_id"]]
    truth = videos["count"].to_numpy()

    grid = load_yaml(args.grid)
    sig_keys, cnt_keys = list(grid["signal"]), list(grid["counter"])
    sig_grid = list(product(*[list(enumerate(v)) for v in grid["signal"].values()]))
    T_vals, dwell_vals = list(enumerate(grid["counter"]["T"])), list(enumerate(grid["counter"]["dwell_s"]))
    edge_vals, round_vals = list(enumerate(grid["counter"]["edge_pause_s"])), list(enumerate(grid["counter"]["rounding"]))
    n_combos = len(sig_grid) * len(T_vals) * len(dwell_vals) * len(edge_vals) * len(round_vals)
    print(f"{len(videos)} videos from {args.splits}; {n_combos} combinations "
          f"({len(sig_grid)} signal x {n_combos // len(sig_grid)} counter)\n")

    rows, keys, params = [], [], []
    t0 = time.time()
    for i, sig_combo in enumerate(sig_grid, 1):
        sig_idx = tuple(j for j, _ in sig_combo)
        sp = SignalParams(**dict(zip(sig_keys, (v for _, v in sig_combo))))
        sigs = signals_for(kps, sp)
        for (ti, T), (di, dwell) in product(T_vals, dwell_vals):
            flips = [detect_flips(*sig, T, dwell) if sig else np.empty(0) for sig in sigs]
            for (ei, edge), (ri, rounding) in product(edge_vals, round_vals):
                cp = CounterParams(T=T, dwell_s=dwell, edge_pause_s=edge, rounding=rounding)
                res = [count_from_flips(f, cp) for f in flips]
                pred = np.array([r.count for r in res])
                half = np.array([len(r.kept) / 2 for r in res])
                row = {**{k: getattr(sp, k) for k in sig_keys}, **cp.to_dict(), **score(pred, truth, half)}
                row["edge_pause_s"] = "off" if edge is None else edge
                rows.append(row)
                keys.append(sig_idx + (ti, di, ei, ri))
                params.append((sp, cp))
        best_so_far = max(r["exact"] for r in rows)
        print(f"[{i:3d}/{len(sig_grid)}] {dict(zip(sig_keys, (v for _, v in sig_combo)))}  "
              f"best exact so far {best_so_far:.1%}  ({time.time() - t0:.0f}s)")

    results = pd.DataFrame(rows)
    results["robust"] = neighbour_mean(keys, results["exact"].to_numpy())
    order = results.sort_values(["exact", "robust", "mae"], ascending=[False, False, True], kind="stable").index
    results = results.loc[order].reset_index(drop=True)
    keys = [keys[i] for i in order]
    best_sp, best_cp = params[order[0]]
    results.to_csv(out_dir / "grid_results.csv", index=False)

    # Baseline vs. best, per video.
    base_res, best_res = run(kps, *BASELINE), run(kps, best_sp, best_cp)
    base_pred = np.array([r.count if r else 0 for r in base_res])
    best_pred = np.array([r.count if r else 0 for r in best_res])
    base_half = np.array([len(r.kept) / 2 if r else 0 for r in base_res])
    best_half = np.array([len(r.kept) / 2 if r else 0 for r in best_res])
    per_video = videos[["video_id", "folder", "count", "duration_s"]].assign(
        baseline=base_pred, best=best_pred, best_half=best_half, best_err=best_pred - truth,
        n_flips=[len(r.flips) if r else 0 for r in best_res],
        n_kept=[len(r.kept) if r else 0 for r in best_res],
        flip_times=[";".join(f"{x:.2f}" for x in r.flips) if r else "no signal" for r in best_res])
    per_video.to_csv(out_dir / "per_video.csv", index=False)

    fmt = {"exact": "{:.1%}".format, "within05": "{:.1%}".format, "within1": "{:.1%}".format,
           "robust": "{:.1%}".format,
           "mae": "{:.2f}".format, "bias": "{:+.2f}".format}
    b = score(base_pred, truth, base_half)
    print(f"\nBaseline (untuned): exact {b['exact']:.1%}, within05 {b['within05']:.1%}, within1 {b['within1']:.1%}, "
          f"mae {b['mae']:.2f}, bias {b['bias']:+.2f}")
    n_tied = (results["exact"] == results["exact"].iloc[0]).sum()
    print(f"\nTop {args.top} of {len(results)} ({n_tied} tie on the best exact score):")
    print(results.head(args.top).to_string(formatters=fmt))

    # One-at-a-time sensitivity around the best: vary one setting, hold the rest.
    print("\nSensitivity around the best (exact accuracy when only this setting changes):")
    best_key = keys[0]
    for d, name in enumerate(sig_keys + cnt_keys):
        same_rest = [i for i, k in enumerate(keys) if all(k[j] == best_key[j] for j in range(len(k)) if j != d)]
        same_rest.sort(key=lambda i: keys[i][d])
        cells = "  ".join(f"{results.at[i, name]}: {results.at[i, 'exact']:.0%}{'*' if keys[i][d] == best_key[d] else ''}"
                          for i in same_rest)
        print(f"  {name:>13}  {cells}")

    wrong = per_video[per_video["best_err"] != 0]
    print(f"\nBest settings get {len(wrong)}/{len(per_video)} videos wrong:")
    if len(wrong):
        print(wrong[["video_id", "folder", "count", "best", "baseline", "n_flips", "n_kept"]].to_string(index=False))
    print(f"\n-> {out_dir / 'grid_results.csv'}\n-> {out_dir / 'per_video.csv'}")

    if args.write:
        s = score(best_pred, truth, best_half)
        frozen = {
            "keypoints": args.keypoints,
            "tuned_on": {"splits": args.splits, "n_videos": len(videos), "date": date.today().isoformat()},
            "signal": asdict(best_sp),
            "counter": best_cp.to_dict(),
            "dev_score": {k: round(v, 4) for k, v in s.items()},
        }
        COUNTER_FILE.write_text("# Frozen by scripts/06_tune_counter.py -- do not edit by hand.\n"
                                + yaml.safe_dump(frozen, sort_keys=False))
        print(f"\nWrote {COUNTER_FILE}")


if __name__ == "__main__":
    main()

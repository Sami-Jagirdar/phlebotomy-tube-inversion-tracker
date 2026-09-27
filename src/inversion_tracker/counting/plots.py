"""Diagnostic figure of one video's orientation signal (used by stages 06a and 07).

    top left      s(t), raw (grey) and smoothed (blue), with the +-T hysteresis thresholds (red);
                  with a count result, also the flips (solid = counted, dotted = dropped edge flip)
    middle left   |v| / L_ref -- dips mean the tube is pointing towards the camera
    bottom left   cap / base / box confidence
    right         every frame's v = cap - base as a point (colour = time), with the PCA axis u
Orange shading marks frames dropped from the signal (low confidence, not interpolated).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from inversion_tracker.counting.signal import Signal
from inversion_tracker.counting.state_machine import CountResult

CAP_COLOR, BASE_COLOR, BOX_COLOR = "tab:red", "tab:blue", "tab:green"
DROP_COLOR = "tab:orange"
FLIP_COLOR = "0.25"


def shade_dropped(ax, t: np.ndarray, keep: np.ndarray) -> None:
    ax.fill_between(t, 0, 1, where=~keep, transform=ax.get_xaxis_transform(),
                    color=DROP_COLOR, alpha=0.2, linewidth=0, step="mid")


def plot_signal(row: pd.Series, sig: Signal, kpt_conf: float, T: float, out_path,
                result: CountResult | None = None) -> None:
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
    if result is not None:
        kept = set(result.kept.tolist())
        for x in result.flips:
            ax_s.axvline(x, color=FLIP_COLOR, lw=1, ls="-" if x in kept else ":")
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
    ax_c.axhline(kpt_conf, color="0.4", ls=":", lw=1)
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

    predicted = "" if result is None else f"predicted = {result.count:g}  |  "
    fig.suptitle(f"video {row.video_id}  |  {row.folder}  |  {row.split}  |  file count = {row['count']}  |  "
                 f"{predicted}{row.duration_s:.1f} s  |  L_ref = {sig.l_ref:.0f} px  |  dropped {1 - keep.mean():.0%}",
                 fontsize=12)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)

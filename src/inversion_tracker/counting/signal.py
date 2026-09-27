"""Per-frame keypoints -> orientation signal s(t).

v(t) = cap - base (crop pixels). s(t) is v projected onto u, the direction v varies along most
(first principal component), measured from the middle of the swing and divided by the tube's full
visible length L_ref:

    s(t) = (v(t) . u - centre) / L_ref

So s is about +-1 at the two ends of a 180-degree swing, about +-0.87 for a 120-degree swing, and
near 0 while the tube points at the camera. The sign is chosen so that + is the side the tube starts on.

This is the offline version: u, centre and L_ref are computed from the whole video.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class SignalParams:
    kpt_conf: float = 0.5    # frames where either keypoint is below this confidence are dropped
    max_gap_s: float = 0.3   # dropped stretches up to this long are linearly interpolated; longer stay NaN
    smooth_s: float = 0.25   # Savitzky-Golay window length
    polyorder: int = 2       # Savitzky-Golay polynomial order
    l_ref_pct: float = 90    # percentile of |v| taken as the full tube length
    centre_pct: float = 5    # swing centre = midpoint of the pct and (100 - pct) percentiles of v . u
    start_s: float = 0.5     # the first this-many seconds of kept frames define the + side


@dataclass
class Signal:
    frames: pd.DataFrame     # t_s, vx, vy, length, s_raw, s, valid, keep, cap_conf, base_conf, box_conf
    axis: np.ndarray         # u, unit vector in crop pixels (y down)
    mean: np.ndarray         # mean of v over kept frames (where the PCA axis is drawn through)
    centre: float            # middle of the swing along u, in pixels
    l_ref: float             # full visible tube length, in pixels
    explained: float         # fraction of v's variance along u (1.0 = perfectly one-dimensional swing)


def fillable(t: np.ndarray, valid: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Frames that are valid, or lie in a gap whose surrounding valid frames are <= max_gap_s apart."""
    tv = t[valid]
    nxt = np.searchsorted(tv, t, side="left")
    prv = np.searchsorted(tv, t, side="right") - 1
    has_both = (prv >= 0) & (nxt < len(tv))
    span = np.full(len(t), np.inf)
    span[has_both] = tv[nxt[has_both]] - tv[prv[has_both]]
    return span <= max_gap_s


def smooth_runs(t: np.ndarray, x: np.ndarray, keep: np.ndarray, window_s: float, polyorder: int) -> np.ndarray:
    """Savitzky-Golay filter applied separately to each contiguous run of kept frames.

    The window is set in frames from the median frame interval (the videos are only mildly VFR).
    Runs shorter than the window are left unsmoothed.
    """
    out = x.copy()
    dt = np.median(np.diff(t))
    win = max(int(round(window_s / dt)) | 1, (polyorder + 2) | 1)  # odd, and longer than polyorder
    edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        if b - a >= win:
            out[a:b] = savgol_filter(x[a:b], win, polyorder)
    return out


def compute_signal(kp: pd.DataFrame, params: SignalParams = SignalParams()) -> Signal:
    """Build s(t) from one video's keypoint table (the output of stage 05)."""
    p = params
    t = kp["t_ms"].to_numpy(float) / 1000
    valid = ((kp["cap_conf"] >= p.kpt_conf) & (kp["base_conf"] >= p.kpt_conf)).to_numpy()  # NaN -> False
    if valid.sum() < 10:
        raise ValueError(f"only {valid.sum()} frames with both keypoints above conf {p.kpt_conf}")
    keep = fillable(t, valid, p.max_gap_s)

    def interp(col: np.ndarray) -> np.ndarray:
        return np.where(keep, np.interp(t, t[valid], col[valid]), np.nan)

    vx = interp((kp["cap_x"] - kp["base_x"]).to_numpy(float))
    vy = interp((kp["cap_y"] - kp["base_y"]).to_numpy(float))
    v = np.column_stack([vx, vy])
    length = np.hypot(vx, vy)

    # PCA on 2-D points: the eigenvector of the covariance matrix with the largest eigenvalue.
    pts = v[keep]
    mean = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(pts, rowvar=False))
    u = evecs[:, -1]
    explained = float(evals[-1] / evals.sum()) if evals.sum() > 0 else 1.0

    proj = v @ u
    lo, hi = np.nanpercentile(proj, [p.centre_pct, 100 - p.centre_pct])
    centre = (lo + hi) / 2  # midrange, so long rests at one end don't pull the centre over
    l_ref = float(np.nanpercentile(length, p.l_ref_pct))
    s_raw = (proj - centre) / l_ref

    t_keep = t[keep]
    start = s_raw[keep][t_keep <= t_keep[0] + p.start_s]
    if np.median(start) < 0:  # make + the starting side
        u, centre, s_raw = -u, -centre, -s_raw

    s = smooth_runs(t, s_raw, keep, p.smooth_s, p.polyorder)
    frames = pd.DataFrame({
        "t_s": t, "vx": vx, "vy": vy, "length": length, "s_raw": s_raw, "s": s,
        "valid": valid, "keep": keep,
        "cap_conf": kp["cap_conf"].to_numpy(), "base_conf": kp["base_conf"].to_numpy(),
        "box_conf": kp["box_conf"].to_numpy(),
    })
    return Signal(frames, u, mean, float(centre), l_ref, explained)

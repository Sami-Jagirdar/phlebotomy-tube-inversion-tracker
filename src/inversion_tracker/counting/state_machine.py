"""Orientation signal s(t) -> inversion count.

A Schmitt trigger (hysteresis) with two states, + and - (the two ends of the swing). The state
switches to - only once s has stayed below -T for dwell_s seconds, and back to + only once it has
stayed above +T for dwell_s. Each switch is a flip (half an inversion); the first state reached is
the starting side and is not a flip. Values between -T and +T never change the state.

One inversion = over and back = 2 flips. By default ("floor") only completed inversions count, so a
trailing one-way move (typically the tube being put down) is not an inversion: 3 flips -> 1.
"none" keeps the half (3 flips -> 1.5), which is useful for diagnosing where a flip was missed or added.
Pickup from the holder and put-down can add a stray flip at either end, so optionally a first/last
flip that is more than edge_pause_s away from its neighbour is dropped (at most one at each end).
"""

import math
from dataclasses import asdict, dataclass

import numpy as np

ROUNDINGS = ("none", "floor", "ceil")


@dataclass(frozen=True)
class CounterParams:
    T: float = 0.5                       # hysteresis threshold, in tube lengths
    dwell_s: float = 0.0                 # time s must stay past a threshold before a flip counts
    edge_pause_s: float | None = None    # drop an isolated first/last flip; None = off
    rounding: str = "floor"              # "floor" = completed inversions only; "ceil"; "none" = flips / 2 (may end in .5)

    def __post_init__(self):
        if self.rounding not in ROUNDINGS:
            raise ValueError(f"rounding must be one of {ROUNDINGS}, got {self.rounding!r}")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CountResult:
    count: float            # whole or half inversions (whole only when rounding is floor/ceil)
    flips: np.ndarray       # times (s) of every flip found by the trigger
    kept: np.ndarray        # flips left after dropping isolated edge flips


def detect_flips(t: np.ndarray, s: np.ndarray, T: float, dwell_s: float) -> np.ndarray:
    """Times (s) at which the state switched; each time is when s first entered the new zone.

    NaN samples (dropped frames) interrupt a pending switch but keep the current state.
    """
    state = 0                    # 0 = not decided yet, +1 / -1 = side of the swing
    cand, cand_t = 0, 0.0        # zone s is currently dwelling in, and since when
    flips = []
    for ti, si in zip(t, s):
        zone = 0 if math.isnan(si) else 1 if si > T else -1 if si < -T else 0
        if zone == 0 or zone == state:
            cand = 0
            continue
        if zone != cand:
            cand, cand_t = zone, ti
        if ti - cand_t >= dwell_s:
            if state != 0:
                flips.append(cand_t)
            state, cand = zone, 0
    return np.array(flips, dtype=float)


def drop_edge_flips(flips: np.ndarray, edge_pause_s: float | None) -> np.ndarray:
    """Drop the first and/or last flip if it is more than edge_pause_s from its neighbour.

    A lone flip has no neighbour, so it is always dropped when this is on.
    """
    if edge_pause_s is None or len(flips) == 0:
        return flips
    if len(flips) == 1:
        return flips[:0]
    start = 1 if flips[1] - flips[0] > edge_pause_s else 0
    stop = len(flips) - 1 if flips[-1] - flips[-2] > edge_pause_s else len(flips)
    return flips[start:max(start, stop)]


def flips_to_count(n_flips: int, rounding: str) -> float:
    if rounding == "none":
        return n_flips / 2
    return n_flips // 2 if rounding == "floor" else (n_flips + 1) // 2


def count_from_flips(flips: np.ndarray, params: CounterParams) -> CountResult:
    kept = drop_edge_flips(flips, params.edge_pause_s)
    return CountResult(flips_to_count(len(kept), params.rounding), flips, kept)


def count_inversions(t: np.ndarray, s: np.ndarray, params: CounterParams = CounterParams()) -> CountResult:
    """Full count for one video from its smoothed signal."""
    return count_from_flips(detect_flips(t, s, params.T, params.dwell_s), params)

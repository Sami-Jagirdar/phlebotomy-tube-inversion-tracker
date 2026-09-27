"""Count accuracy scores, shared by tuning (stage 06) and evaluation (stage 07)."""

import numpy as np


def score(pred: np.ndarray, truth: np.ndarray, half: np.ndarray | None = None) -> dict:
    """Scores of the (rounded) prediction against the file-name count.

    exact     pred == truth
    within05  the unrounded half count (counted flips / 2) is within 0.5 of truth, 0.5 included --
              e.g. truth 2, 3 flips (1.5) counts, even though floor rounding predicts 1.
              Without `half`, the prediction itself is used.
    within1   |pred - truth| <= 1
    mae, bias mean |pred - truth| and mean (pred - truth), in inversions
    """
    pred, truth = np.asarray(pred, dtype=float), np.asarray(truth, dtype=float)
    half = pred if half is None else np.asarray(half, dtype=float)
    err = pred - truth
    return {"exact": float(np.mean(err == 0)), "within05": float(np.mean(np.abs(half - truth) <= 0.5)),
            "within1": float(np.mean(np.abs(err) <= 1)),
            "mae": float(np.mean(np.abs(err))), "bias": float(np.mean(err))}

"""Shared rendering helpers for the demo scripts (tools/demo_video.py, tools/demo_video_sidebyside.py).

Frame colours match tools/preview_video.py; the dropped-frame tint matches the orange shading in
counting/plots.py.
"""

import cv2
import numpy as np
import pandas as pd

BOX_COLOR = (0, 255, 0)       # green
CAP_COLOR = (0, 0, 255)       # red
BASE_COLOR = (255, 0, 0)      # blue
ARROW_COLOR = (255, 255, 255)
TEXT_COLOR = (0, 255, 255)    # yellow
DROP_TINT = (0, 165, 255)     # orange


def running_count(t: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Completed-inversion count visible at each frame time.

    `kept` is CountResult.kept (edge-cleaned flip times, chronological). This matches the final
    count only under "floor" rounding, i.e. the frozen configs/counter.yaml.
    """
    return np.searchsorted(kept, t, side="right") // 2


def draw_point(frame: np.ndarray, pt: tuple[int, int], color, visible: bool) -> None:
    cv2.circle(frame, pt, 9, color, -1 if visible else 3, cv2.LINE_AA)


def annotate_frame(frame: np.ndarray, kp_row: pd.Series, kpt_conf: float, keep: bool,
                    s_value: float, T: float, count: float) -> np.ndarray:
    """Draw the detection, signal readout and running count onto one frame. Returns the frame."""
    frame = np.ascontiguousarray(frame)
    if pd.isna(kp_row.box_conf):
        status = "no detection"
    else:
        x1, y1, x2, y2 = (int(round(v)) for v in (kp_row.box_x1, kp_row.box_y1, kp_row.box_x2, kp_row.box_y2))
        cx, cy = int(round(kp_row.cap_x)), int(round(kp_row.cap_y))
        bx, by = int(round(kp_row.base_x)), int(round(kp_row.base_y))
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.arrowedLine(frame, (bx, by), (cx, cy), ARROW_COLOR, 2, cv2.LINE_AA, tipLength=0.08)
        draw_point(frame, (cx, cy), CAP_COLOR, kp_row.cap_conf >= kpt_conf)
        draw_point(frame, (bx, by), BASE_COLOR, kp_row.base_conf >= kpt_conf)
        status = f"box {kp_row.box_conf:.2f}  cap {kp_row.cap_conf:.2f}  base {kp_row.base_conf:.2f}"

    if not keep:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), DROP_TINT, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        status += "  (dropped)"

    cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)
    s_text = "s = nan" if np.isnan(s_value) else f"s = {s_value:+.2f}  (T = {T:.2f})"
    cv2.putText(frame, s_text, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, TEXT_COLOR, 2, cv2.LINE_AA)

    count_text = f"count: {count:g}"
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 3)
    cv2.putText(frame, count_text, (frame.shape[1] - tw - 25, th + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 3, cv2.LINE_AA)
    return frame


class SignalPanel:
    """A s(t) plot that reveals itself frame by frame, for the side-by-side demo video.

    Static elements (axes, thresholds) are drawn once; render() only updates the revealed line,
    the current-position dot, and any newly-passed flip markers, then rasterises the figure.
    """

    def __init__(self, t: np.ndarray, s: np.ndarray, T: float, flips: np.ndarray, kept: set,
                 height_px: int, width_px: int, dpi: int = 100):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.t, self.s, self.flips, self.kept = t, s, flips, kept
        self.fig, self.ax = plt.subplots(figsize=(width_px / dpi, height_px / dpi), dpi=dpi, layout="constrained")
        self.ax.set_xlim(t[0], t[-1])
        self.ax.set_ylim(-1.4, 1.4)
        self.ax.axhline(0, color="0.5", lw=0.8)
        for y in (T, -T):
            self.ax.axhline(y, color="tab:red", ls="--", lw=1)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("s(t)   (+ = start side)")
        self.ax.grid(alpha=0.3)
        (self.line,) = self.ax.plot([], [], color="tab:blue", lw=1.8)
        (self.dot,) = self.ax.plot([], [], "o", color="tab:blue", ms=7)
        self.count_text = self.ax.text(0.02, 0.96, "", transform=self.ax.transAxes,
                                        fontsize=13, va="top", fontweight="bold")
        self.n_drawn_flips = 0

    def render(self, i: int, count: float) -> np.ndarray:
        t, s = self.t[: i + 1], self.s[: i + 1]
        self.line.set_data(t, s)
        if not np.isnan(s[-1]):
            self.dot.set_data([t[-1]], [s[-1]])
        n_due = int(np.searchsorted(self.flips, self.t[i], side="right"))
        while self.n_drawn_flips < n_due:
            x = self.flips[self.n_drawn_flips]
            self.ax.axvline(x, color="0.25", lw=1, ls="-" if x in self.kept else ":")
            self.n_drawn_flips += 1
        self.count_text.set_text(f"count: {count:g}")
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())
        return cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR)

    def close(self) -> None:
        import matplotlib.pyplot as plt
        plt.close(self.fig)

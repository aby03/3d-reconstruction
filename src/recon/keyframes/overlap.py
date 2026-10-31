"""Image overlap with the current keyframe: KLT tracks of its corners, chained frame to frame."""

from collections.abc import Sequence

import cv2
import numpy as np

KLT_LONG_SIDE = 480  # tracking resolution
MAX_POINTS = 500
WIN_SIZE = (21, 21)
MAX_LEVEL = 2  # 3 pyramid levels
FB_MAX_PX = 1.0  # forward-backward error above this drops a track


class KeyframeTracker:
    """Seeds Shi-Tomasi corners on a keyframe and follows them through the next frames, one frame at a time.

    `grays` are uint8 gray frames indexed by frame idx, all the same size; None marks an undecodable frame.
    """

    def __init__(self, grays: Sequence[np.ndarray | None]):
        self.grays = grays
        shape = next(g.shape for g in grays if g is not None)
        self.diag = float(np.hypot(*shape))
        self.prev: np.ndarray | None = None
        self.points = np.empty((0, 1, 2), np.float32)  # current positions of the surviving tracks
        self.origin = np.empty((0, 1, 2), np.float32)  # their positions on the keyframe
        self.num_seeded = 0

    def seed(self, idx: int) -> None:
        self.prev = self.grays[idx]
        points = None
        if self.prev is not None:
            points = cv2.goodFeaturesToTrack(self.prev, MAX_POINTS, qualityLevel=0.01, minDistance=7, blockSize=7)
        self.points = np.empty((0, 1, 2), np.float32) if points is None else points.astype(np.float32)
        self.origin = self.points.copy()
        self.num_seeded = len(self.points)

    def step(self, idx: int) -> tuple[float, float]:
        """Track into frame `idx` (the frame after the previous step); returns (flow_med, track_ratio).

        flow_med is the median displacement from the keyframe in units of the image diagonal (NaN without tracks),
        track_ratio the fraction of seeded corners still tracked.
        """
        current = self.grays[idx]
        if current is None or self.prev is None or not len(self.points):
            self.points, self.origin, self.prev = self.points[:0], self.origin[:0], current
            return np.nan, 0.0
        lk = {"winSize": WIN_SIZE, "maxLevel": MAX_LEVEL}
        forward, st_f, _ = cv2.calcOpticalFlowPyrLK(self.prev, current, self.points, None, **lk)
        back, st_b, _ = cv2.calcOpticalFlowPyrLK(current, self.prev, forward, None, **lk)
        fb_error = np.linalg.norm((back - self.points).reshape(-1, 2), axis=1)
        good = (st_f.ravel() == 1) & (st_b.ravel() == 1) & (fb_error < FB_MAX_PX)
        self.points, self.origin, self.prev = forward[good], self.origin[good], current
        if not len(self.points):
            return np.nan, 0.0
        flow = np.linalg.norm((self.points - self.origin).reshape(-1, 2), axis=1)
        return float(np.median(flow)) / self.diag, len(self.points) / self.num_seeded

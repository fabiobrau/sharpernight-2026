"""Mode B: find the 4 ArUco markers and warp the chosen poster into the board.

The critical rule is in app.py, not here: the composited frame is what gets fed
to YOLO. If you ever detect first and paste afterwards you have built a fake
demo. This module only produces the composited frame.

The markers sit at the four corners and we warp into the quad bounded by their
*inner* corners, so the poster never covers a marker and the board keeps
tracking frame after frame.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger("invisible.compositor")

# Board corner -> marker id. Printed board must use the same mapping.
ID_TL, ID_TR, ID_BR, ID_BL = 0, 1, 2, 3
BOARD_IDS = (ID_TL, ID_TR, ID_BR, ID_BL)

# For each board corner, which corner of that marker bounds the inner region.
# aruco returns each marker's corners clockwise from its own top-left.
_INNER_CORNER = {ID_TL: 2, ID_TR: 3, ID_BR: 0, ID_BL: 1}

ARUCO_DICT = cv2.aruco.DICT_4X4_50


class BoardTracker:
    """Finds the marker board and hands back a smoothed inner quad."""

    def __init__(self, smooth: float = 0.5, miss_tolerance: int = 4) -> None:
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        params = cv2.aruco.DetectorParameters()
        # Sub-pixel corners keep the warp from shimmering under the child's hands.
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self._detector = cv2.aruco.ArucoDetector(dictionary, params)

        self.smooth = smooth
        self.miss_tolerance = miss_tolerance
        self._quad: np.ndarray | None = None
        self._misses = 0

    @property
    def found(self) -> bool:
        return self._quad is not None

    def update(self, frame: np.ndarray) -> np.ndarray | None:
        """Return the 4x2 inner quad (TL, TR, BR, BL) or None."""
        try:
            corners, ids, _ = self._detector.detectMarkers(frame)
        except Exception as exc:  # a bad frame must not kill the loop
            log.debug("aruco failed: %s", exc)
            corners, ids = (), None

        quad = self._quad_from(corners, ids)
        if quad is None:
            self._misses += 1
            if self._misses > self.miss_tolerance:
                self._quad = None
            return self._quad

        self._misses = 0
        if self._quad is None:
            self._quad = quad
        else:
            a = self.smooth
            self._quad = a * self._quad + (1.0 - a) * quad
        return self._quad

    def _quad_from(self, corners, ids) -> np.ndarray | None:
        if ids is None or len(ids) < 4:
            return None
        found: dict[int, np.ndarray] = {}
        for c, i in zip(corners, ids.flatten()):
            i = int(i)
            if i in _INNER_CORNER:
                found[i] = c.reshape(4, 2)
        if not all(i in found for i in BOARD_IDS):
            return None
        return np.array(
            [found[i][_INNER_CORNER[i]] for i in BOARD_IDS], dtype=np.float32)


def warp_into(frame: np.ndarray, patch: np.ndarray,
              quad: np.ndarray, feather: int = 3) -> np.ndarray:
    """Composite `patch` into the quad of `frame` and return the new frame.

    Only the quad's bounding box is touched. Warping and alpha-blending the
    whole 1280x720 canvas cost ~25 ms per frame, which alone would have put the
    demo under the 20 fps target.
    """
    h, w = frame.shape[:2]
    dst = quad.astype(np.float32)

    pad = feather * 3 + 2
    x0 = max(0, int(np.floor(dst[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(dst[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(dst[:, 0].max())) + pad)
    y1 = min(h, int(np.ceil(dst[:, 1].max())) + pad)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return frame
    rw, rh = x1 - x0, y1 - y0

    ph, pw = patch.shape[:2]
    src = np.array([[0, 0], [pw - 1, 0], [pw - 1, ph - 1], [0, ph - 1]], np.float32)
    local = dst - np.array([x0, y0], np.float32)
    try:
        M = cv2.getPerspectiveTransform(src, local)
    except cv2.error as exc:
        log.debug("degenerate quad: %s", exc)
        return frame

    warped = cv2.warpPerspective(patch, M, (rw, rh), flags=cv2.INTER_LINEAR)
    mask = np.zeros((rh, rw), np.uint8)
    cv2.fillConvexPoly(mask, local.astype(np.int32), 255)
    if feather > 0:
        k = feather * 2 + 1
        mask = cv2.GaussianBlur(mask, (k, k), 0)

    out = frame.copy()
    roi = out[y0:y1, x0:x1]
    alpha = (mask.astype(np.float32) / 255.0)[:, :, None]
    roi[:] = (roi.astype(np.float32) * (1 - alpha)
              + warped.astype(np.float32) * alpha).astype(np.uint8)
    return out


def draw_board_outline(frame: np.ndarray, quad: np.ndarray,
                       color=(80, 230, 120), thickness: int = 3) -> None:
    cv2.polylines(frame, [quad.astype(np.int32)], True, color, thickness, cv2.LINE_AA)


def board_inner_frac(margin_frac: float = 0.02,
                     marker_frac: float = 0.14) -> tuple[float, float]:
    """Where the poster sits inside the board, as fractions of the board side.

    The markers eat into the poster area, and poster area is attack strength, so
    they are kept as small as detection allows. Returns (start, end).
    """
    return margin_frac + marker_frac, 1.0 - (margin_frac + marker_frac)


def make_board_image(px: int = 2480, margin_frac: float = 0.02,
                     marker_frac: float = 0.14) -> np.ndarray:
    """Render the printable marker board (white, 4 markers, empty middle)."""
    img = np.full((px, px, 3), 255, np.uint8)
    dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
    m = int(px * marker_frac)
    off = int(px * margin_frac)
    positions = {
        ID_TL: (off, off),
        ID_TR: (px - off - m, off),
        ID_BR: (px - off - m, px - off - m),
        ID_BL: (off, px - off - m),
    }
    for mid, (x, y) in positions.items():
        marker = cv2.aruco.generateImageMarker(dictionary, mid, m)
        img[y:y + m, x:x + m] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    return img

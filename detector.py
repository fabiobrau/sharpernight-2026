"""Ultralytics wrapper: MPS inference, largest-person tracking, flicker control.

Two things live here:
  * `Detector`  -- one YOLO model, one `detect()` call, never raises at runtime.
  * `StableTracker` -- turns noisy per-frame detections into the steady
    "is there a human right now?" signal the show depends on.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("invisible.detector")


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int = 0

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def scaled(self, sx: float, sy: float) -> "Detection":
        return Detection(self.x1 * sx, self.y1 * sy, self.x2 * sx, self.y2 * sy,
                         self.conf, self.cls)


class ModelMismatch(RuntimeError):
    """Raised at startup when the patch was not made for the loaded weights."""


class Detector:
    def __init__(self, weights: str, device: str = "mps", imgsz: int = 640,
                 conf: float = 0.4, classes: list[int] | None = None,
                 name: str = "") -> None:
        from ultralytics import YOLO  # imported late: keeps --help instant

        if not os.path.exists(weights):
            raise FileNotFoundError(
                f"model weights not found: {weights}\n"
                "Pre-download them before the fair -- the venue wifi will fail.")
        self.name = name or os.path.splitext(os.path.basename(weights))[0]
        self.imgsz = imgsz
        self.conf = conf
        self.classes = classes if classes is not None else [0]
        self.device = device
        self.model = YOLO(weights)
        self._fail_count = 0
        self._calls = 0

    def warmup(self, shape: tuple[int, int] = (720, 1280)) -> None:
        """Run one throwaway inference so the first child does not wait."""
        blank = np.zeros((shape[0], shape[1], 3), np.uint8)
        try:
            self.detect(blank)
        except Exception as exc:  # pragma: no cover - warmup must never block
            log.warning("warmup failed (%s); continuing", exc)

    def detect(self, frame: np.ndarray, classes: list[int] | None = None,
               conf: float | None = None) -> list[Detection]:
        """Return the boxes. On failure, return [] and keep the show alive."""
        try:
            res = self.model.predict(
                frame, imgsz=self.imgsz,
                conf=self.conf if conf is None else conf,
                classes=self.classes if classes is None else classes,
                device=self.device, verbose=False)[0]
            self._fail_count = 0
            # Torch's MPS allocator creeps upward over a long session (~4 MB per
            # 1000 frames). Harmless in a short run, ~1 GB over a three-hour
            # afternoon, so evict periodically. Cheap: once every ~20 s.
            self._calls += 1
            if self._calls % 600 == 0:
                try:
                    import torch
                    torch.mps.empty_cache()
                except Exception:
                    pass
        except Exception as exc:
            self._fail_count += 1
            if self._fail_count in (1, 10, 100):
                log.error("inference failed (%dx): %s", self._fail_count, exc)
            return []

        out: list[Detection] = []
        for b in res.boxes:
            x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
            out.append(Detection(x1, y1, x2, y2, float(b.conf[0]), int(b.cls[0])))
        return out

    def largest(self, frame: np.ndarray, min_area: float = 0.0) -> Detection | None:
        """Only the biggest person matters -- ignore the queue behind the child."""
        dets = [d for d in self.detect(frame)
                if d.area >= min_area and d.cls == 0]
        return max(dets, key=lambda d: d.area) if dets else None


class StableTracker:
    """Smooths detections into a stable visible/invisible state.

    A detection is held for `hold_frames` empty frames before we admit the human
    is gone, and the "vanished" celebration only fires after `vanish_seconds` of
    continuous non-detection. Together these kill the flicker that would
    otherwise fire the fanfare every few frames.
    """

    def __init__(self, hold_frames: int = 3, vanish_seconds: float = 0.6,
                 refound_frames: int = 2) -> None:
        self.hold_frames = hold_frames
        self.vanish_seconds = vanish_seconds
        self.refound_frames = refound_frames

        self.box: Detection | None = None   # last box worth drawing
        self.visible = False                # smoothed "human is detected"
        self.conf = 0.0                     # smoothed confidence, for the bar

        self._empty_frames = 0
        self._hit_frames = 0
        self._lost_since: float | None = None

        # One-shot events the show reacts to.
        self.just_vanished = False
        self.just_found = False
        self._vanish_fired = False

    def reset(self) -> None:
        self.__init__(self.hold_frames, self.vanish_seconds, self.refound_frames)

    def update(self, det: Detection | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.just_vanished = False
        self.just_found = False

        if det is not None:
            self._empty_frames = 0
            self._hit_frames += 1
            self.box = det
            # Confidence chases the truth fast -- kids notice lag and lose the
            # causal link between poster and disappearance.
            self.conf = det.conf if self.conf == 0.0 else 0.5 * self.conf + 0.5 * det.conf
            if not self.visible and self._hit_frames >= self.refound_frames:
                self.visible = True
                if self._vanish_fired:
                    self.just_found = True
                self._vanish_fired = False
                self._lost_since = None
            return

        # No detection this frame.
        self._hit_frames = 0
        self._empty_frames += 1
        self.conf = max(0.0, self.conf * 0.45)  # bar drains visibly

        if self._empty_frames <= self.hold_frames:
            return  # still holding the old box: probably just a blink

        if self.visible:
            self.visible = False
            self._lost_since = now
        self.box = None

        if (self._lost_since is not None and not self._vanish_fired
                and now - self._lost_since >= self.vanish_seconds):
            self._vanish_fired = True
            self.just_vanished = True

    @property
    def invisible_seconds(self) -> float:
        """How long the child has been gone, counted from the moment of loss."""
        if self.visible or self._lost_since is None:
            return 0.0
        return max(0.0, time.monotonic() - self._lost_since)

    @property
    def celebrating(self) -> bool:
        return self._vanish_fired and not self.visible

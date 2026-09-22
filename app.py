"""INVISIBLE! -- can you hide from the robot?

A science-fair demo: a YOLO detector boxes the user, the user raises a poster,
the box pops like a soap bubble, and a timer counts how long they stayed
invisible.

    python app.py --lang it

THE ONE RULE: in virtual mode the poster is warped into the frame BEFORE the
frame reaches the detector. Detecting first and pasting afterwards would be a
fake demo. See `Show._process`.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Some MPS kernels still fall back to CPU; set this before torch is imported.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import yaml

import overlay
from audio import Audio
from compositor import BoardTracker, draw_board_outline, warp_into
from detector import Detection, Detector, ModelMismatch, StableTracker
from scoreboard import Scoreboard
from strings import Strings, poster_label

ROOT = os.path.dirname(os.path.abspath(__file__))
WND = "INVISIBLE!"

CANVAS_W, CANVAS_H = 1600, 900
PANEL_W, PANEL_H = 800, 450
BRAND_H = 78                        # top strip: logos left, event name right
HEAD_H = 52                         # panel titles
PANEL_Y = BRAND_H + HEAD_H          # top of the two camera panels
BAR_Y = PANEL_Y + PANEL_H           # top of the bottom bar

log = logging.getLogger("invisible")

POSTER_KEYS = {ord("1"): "adversarial", ord("2"): "dog", ord("0"): "none"}


# --------------------------------------------------------------- camera ---

class Camera:
    """Webcam with automatic reconnection. Never raises at read time."""

    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index = index
        self.width, self.height, self.fps = width, height, fps
        self.cap: cv2.VideoCapture | None = None
        self.failures = 0
        self._next_retry = 0.0
        self.open()

    def open(self) -> bool:
        self.release()
        try:
            cap = cv2.VideoCapture(self.index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            if not cap.isOpened():
                cap.release()
                return False
            self.cap = cap
            self.failures = 0
            log.info("camera %d opened (%dx%d)", self.index,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            return True
        except Exception as exc:
            log.error("camera open failed: %s", exc)
            return False

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            if time.monotonic() >= self._next_retry:
                self._next_retry = time.monotonic() + 2.0
                self.open()
            return None
        try:
            ok, frame = self.cap.read()
        except Exception as exc:
            log.error("camera read raised: %s", exc)
            ok, frame = False, None
        if not ok or frame is None:
            self.failures += 1
            if self.failures > 15:
                log.warning("camera lost; reconnecting")
                self.release()
                self._next_retry = time.monotonic() + 1.0
            return None
        self.failures = 0
        # NOTE: no flip here. Mirroring is a *display* transform applied in
        # Show._process; the detector must see the world as the camera does, or
        # a printed patch reaches it horizontally flipped and the attack
        # weakens (Mode A).
        return frame

    @property
    def alive(self) -> bool:
        return self.cap is not None

    def release(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class FallbackSource:
    """Mode C: loop a recorded clip, or cycle stills. Never shows a traceback."""

    def __init__(self, video: str, stills_dir: str) -> None:
        self.cap = None
        self.stills: list[np.ndarray] = []
        self._i = 0
        self._last = 0.0
        if os.path.exists(video):
            cap = cv2.VideoCapture(video)
            if cap.isOpened():
                self.cap = cap
                log.info("fallback clip: %s", video)
        if os.path.isdir(stills_dir):
            for fn in sorted(os.listdir(stills_dir)):
                img = cv2.imread(os.path.join(stills_dir, fn))
                if img is not None:
                    self.stills.append(img)
            if self.stills:
                log.info("fallback stills: %d", len(self.stills))

    @property
    def available(self) -> bool:
        return self.cap is not None or bool(self.stills)

    def read(self) -> np.ndarray | None:
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self.cap.read()
            if ok:
                return frame
        if self.stills:
            now = time.monotonic()
            if now - self._last > 3.0:
                self._last = now
                self._i = (self._i + 1) % len(self.stills)
            return self.stills[self._i].copy()
        return None


# ------------------------------------------------------------------ show ---

class Show:
    def __init__(self, cfg: dict, args: argparse.Namespace) -> None:
        self.cfg = cfg
        self.args = args
        self.S = Strings(args.lang or cfg["ui"]["lang"])
        self.colors = {k: tuple(v) for k, v in cfg["ui"]["colors"].items()}
        overlay.set_font(os.path.join(ROOT, cfg["ui"]["font"]))

        self.mode = args.mode
        self.mirror = bool(cfg["camera"]["mirror"])
        trg = cfg.get("trigger", {})
        self.trigger = args.trigger or str(trg.get("mode", "detector"))
        self._sim_fade = float(trg.get("fade_seconds", 0.35))
        self._sim_level = 1.0          # 1 = detector untouched, 0 = fully suppressed
        self._force_vanish = False     # manual override, any mode (V)
        self._trig_class = int(trg.get("target_class", 78))
        self._trig_conf = float(trg.get("threshold", 0.35))
        self._trig_latch = int(trg.get("latch_frames", 12))
        self._trig_score = 0.0         # live strength of the trigger class
        self._trig_since = 10**9       # frames since the class last fired
        self.poster = cfg["patch"]["active"]
        self.expert = False
        self.fullscreen = bool(cfg["ui"]["fullscreen"])

        m = cfg["model"]
        self.detector = Detector(os.path.join(ROOT, m["weights"]), m["device"],
                                 m["imgsz"], float(m["conf"]), list(m["classes"]),
                                 name=m["name"])
        self.expert_detector = None
        if cfg["expert"]["enabled"] and os.path.exists(
                os.path.join(ROOT, cfg["expert"]["weights"])):
            try:
                self.expert_detector = Detector(
                    os.path.join(ROOT, cfg["expert"]["weights"]), m["device"],
                    m["imgsz"], float(m["conf"]), list(m["classes"]),
                    name=cfg["expert"]["name"])
            except Exception as exc:
                log.warning("expert model unavailable: %s", exc)

        st = cfg["stability"]
        self.tracker = StableTracker(int(st["hold_frames"]),
                                     float(st["vanish_seconds"]),
                                     int(st["refound_frames"]))
        self.board = BoardTracker()
        self.mascot = overlay.Mascot(os.path.join(ROOT, "assets/sprites"))
        brand = cfg.get("branding", {})
        entries = []
        for e in brand.get("logos", []) or []:
            e = dict(e)
            e["path"] = os.path.join(ROOT, e.get("path", ""))
            entries.append(e)
        self.branding = overlay.Branding(entries, str(brand.get("text", "")))
        self.puff = overlay.PuffBurst()
        self.audio = Audio(os.path.join(ROOT, "assets/sounds"),
                           bool(cfg["sound"]["enabled"]) and not args.mute,
                           float(cfg["sound"]["volume"]))
        self.scores = Scoreboard(int(cfg["scoreboard"]["size"]),
                                 float(cfg["scoreboard"]["min_time_seconds"]))

        self.patches = self._load_patches()
        if self.trigger == "neuron":
            log.warning("NEURON TRIGGER: the box is fired by class %d rising "
                        "above %.2f, not by the detector losing the user.",
                        self._trig_class, self._trig_conf)
        self.camera: Camera | None = None
        self.fallback = FallbackSource(os.path.join(ROOT, cfg["fallback"]["video"]),
                                       os.path.join(ROOT, cfg["fallback"]["stills_dir"]))

        self._frozen_time = 0.0
        self._frozen_until = 0.0
        self._is_record = False
        self._expert_box: Detection | None = None
        self._last_box: Detection | None = None
        self._candidates: list[Detection] = []
        self._last_frame_w = 1280
        self._last_frame: np.ndarray | None = None
        self._last_frame_at = 0.0
        self._canvas = np.full((CANVAS_H, CANVAS_W, 3),
                               self.colors["panel_bg"], np.uint8)
        self._frame_no = 0
        self._fps = 0.0
        self._t_last = time.monotonic()
        self._running = True

    # -- assets ------------------------------------------------------------
    def _load_patches(self) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, rel in self.cfg["patch"]["images"].items():
            p = os.path.join(ROOT, rel)
            img = cv2.imread(p)
            if img is None:
                log.warning("poster image missing: %s", p)
            else:
                out[key] = img
        if "adversarial" not in out:
            log.error("the adversarial patch is missing -- the demo cannot work")
        return out

    # -- main loop ---------------------------------------------------------
    def run(self) -> int:
        if self.mode != "fallback":
            cam = self.cfg["camera"]
            self.camera = Camera(int(self.args.camera if self.args.camera is not None
                                     else cam["index"]),
                                 int(cam["width"]), int(cam["height"]),
                                 int(cam["fps"]))
        self.detector.warmup()
        if self.expert_detector:
            self.expert_detector.warmup()

        cv2.namedWindow(WND, cv2.WINDOW_NORMAL)
        self._apply_fullscreen()

        while self._running:
            try:
                self._tick()
            except KeyboardInterrupt:
                break
            except Exception as exc:  # the show must never die in front of an audience
                log.exception("frame failed, continuing: %s", exc)
                time.sleep(0.05)

        self.shutdown()
        return 0

    def _tick(self) -> None:
        frame = self.camera.read() if self.camera else None
        using_fallback = False
        now = time.monotonic()

        if frame is not None:
            self._last_frame = frame
            self._last_frame_at = now
        else:
            # A webcam drops the odd frame. Riding it out on the last good one
            # beats flashing a "camera lost" screen at a seven-year-old.
            if (self._last_frame is not None
                    and now - self._last_frame_at < 1.0):
                frame = self._last_frame
            else:
                fb = self.fallback.read()
                if fb is not None:
                    frame, using_fallback = fb, True

        if frame is None:
            canvas = self._compose_waiting()
        else:
            clean, robot = self._process(frame)
            canvas = self._compose(clean, robot, using_fallback)
        cv2.imshow(WND, canvas)

        self._handle_keys()

        self._frame_no += 1
        now = time.monotonic()
        dt = now - self._t_last
        self._t_last = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt) if self._fps else 1.0 / dt

    # -- the important part ------------------------------------------------
    def _process(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (what the user sees, what the detector was given).

        In virtual mode the poster is warped in FIRST, and that composited frame
        is what YOLO sees. The left-hand panel shows the untouched camera so the
        audience can tell the user never actually disappeared.
        """
        clean = frame
        detect_frame = frame
        quad = None

        if self.mode == "virtual":
            quad = self.board.update(frame)
            patch = self.patches.get(self.poster) if self.poster != "none" else None
            if quad is not None and patch is not None:
                detect_frame = warp_into(frame, patch, quad)

        h, w = detect_frame.shape[:2]
        self._last_frame_w = w
        min_area = float(self.cfg["stability"]["min_box_area_frac"]) * w * h
        if self.trigger == "neuron":
            # One pass gets both: the person, and the class the patch lights up.
            # The trigger class is read at a low floor so the bar can respond
            # before the class is confident enough to count as fired.
            allb = self.detector.detect(detect_frame,
                                        classes=[0, self._trig_class], conf=0.10)
            persons = [b for b in allb
                       if b.cls == 0 and b.area >= min_area and b.conf >= self.detector.conf]
            self._candidates = [b for b in allb if b.cls == self._trig_class]
            det = max(persons, key=lambda b: b.area) if persons else None
        else:
            self._candidates = []
            det = self.detector.largest(detect_frame, min_area)
        det = self._maybe_suppress(det, quad)
        self.tracker.update(det)

        if self.expert and self.expert_detector and self._frame_no % 3 == 0:
            self._expert_box = self.expert_detector.largest(detect_frame, min_area)

        self._react()

        # warp_into already returned a fresh array; only copy when it didn't run,
        # so we never draw the overlay onto the clean camera frame.
        robot = detect_frame if detect_frame is not frame else frame.copy()
        if quad is not None and self.args.show_board:
            draw_board_outline(robot, quad, self.colors["good"], 3)

        # Mirror for display only, and mirror the boxes to match, so the user
        # sees themselves the right way round and no overlay text comes out
        # backwards.
        if self.mirror:
            clean = cv2.flip(clean, 1)
            robot = cv2.flip(robot, 1)
        self._draw_robot_view(robot)
        return clean, robot

    def _trigger_fired(self) -> bool:
        """Has poster A's output class lit up?

        The patch was optimised to drive exactly this neuron, so reading it is a
        single comparison on detections we already have. Nothing else in the room
        produces it -- not the decoy, not a hand, not an empty board.
        """
        if self.trigger != "neuron":
            return False
        self._trig_score = max((b.conf for b in self._candidates
                                if b.cls == self._trig_class), default=0.0)
        if self._trig_score >= self._trig_conf:
            self._trig_since = 0
        else:
            self._trig_since += 1
        return self._trig_since <= self._trig_latch

    def _maybe_suppress(self, det: Detection | None,
                        quad) -> Detection | None:
        """Wizard-of-Oz suppression. Returns the detection, weakened or dropped.

        Only ever active when simulation is switched on, or the operator is
        holding the manual override. With both off this is a no-op and the demo
        shows exactly what the detector reports.
        """
        want = self._force_vanish or self._trigger_fired()

        # Ease the confidence down instead of cutting it: an instant drop to
        # zero looks wrong next to a real detector, and kids notice.
        step = 1.0 / max(1e-3, self._sim_fade * max(self._fps, 1.0))
        target = 0.0 if want else 1.0
        if self._sim_level < target:
            self._sim_level = min(target, self._sim_level + step)
        elif self._sim_level > target:
            self._sim_level = max(target, self._sim_level - step)

        if det is None or self._sim_level >= 1.0:
            return det
        weakened = Detection(det.x1, det.y1, det.x2, det.y2,
                             det.conf * self._sim_level)
        return weakened if weakened.conf >= float(self.cfg["model"]["conf"]) else None

    def _react(self) -> None:
        """Turn tracker events into sound, particles and scores."""
        t = self.tracker
        if t.just_vanished:
            self.audio.play("pop")
            self.audio.play("confused")
            if t.box is None and self._last_box is not None:
                b = self._last_box
                cx, cy = b.center
                if self.mirror:
                    cx = self._last_frame_w - cx
                self.puff.fire(cx, cy, max(b.x2 - b.x1, b.y2 - b.y1))
        if t.just_found:
            seconds = self._frozen_time
            self._frozen_until = time.monotonic() + 3.0
            self._is_record = False
            if seconds >= float(self.cfg["scoreboard"]["min_time_seconds"]):
                self.scores.submit(seconds)
                self._is_record = self.scores.is_new_best(seconds)
            self.audio.play("fanfare")
        if t.visible and t.box is not None:
            self._last_box = t.box
        if t.celebrating:
            self._frozen_time = t.invisible_seconds

    # -- drawing -----------------------------------------------------------
    def _disp(self, box, w: int) -> tuple[float, float, float, float]:
        """Detector coordinates -> display coordinates (mirrored if configured)."""
        if not self.mirror:
            return box.x1, box.y1, box.x2, box.y2
        return w - box.x2, box.y1, w - box.x1, box.y2

    def _draw_robot_view(self, img: np.ndarray) -> None:
        t = self.tracker
        w = img.shape[1]
        if t.box is not None and t.visible:
            x1, y1, x2, y2 = self._disp(t.box, w)
            overlay.corner_box(img, (x1, y1, x2, y2), self.colors["box"], 10)
            label = f"{self.S.get('label_human')} {int(round(t.box.conf * 100))}%"
            overlay.draw_text(img, label, x1, y1 - 12, 40,
                              self.colors["box"], "bl", 5)
        if self.expert and self._expert_box is not None:
            ex1, ey1, ex2, ey2 = self._disp(self._expert_box, w)
            overlay.corner_box(img, (ex1, ey1, ex2, ey2), (255, 120, 220), 6,
                               frac=0.20)
            # Keep the label on screen when the box runs to the bottom edge.
            ly, anchor = ey2 + 10, "tl"
            if ly + 40 > img.shape[0]:
                ly, anchor = ey2 - 10, "bl"
            overlay.draw_text(img, self.S.get("expert_model"), ex1, ly, 30,
                              (255, 120, 220), anchor, 4)
        self.puff.update_and_draw(img)

    def _compose(self, clean: np.ndarray, robot: np.ndarray,
                 using_fallback: bool) -> np.ndarray:
        # One canvas for the whole session: allocating 1600x900x3 per frame cost
        # ~10 ms, and the panels overwrite themselves anyway.
        canvas = self._canvas
        canvas[0:PANEL_Y] = self.colors["panel_bg"]
        left = cv2.resize(clean, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        right = cv2.resize(robot, (PANEL_W, PANEL_H), interpolation=cv2.INTER_LINEAR)
        canvas[PANEL_Y:PANEL_Y + PANEL_H, 0:PANEL_W] = left
        canvas[PANEL_Y:PANEL_Y + PANEL_H, PANEL_W:PANEL_W * 2] = right

        overlay.draw_text(canvas, self.S.get("panel_human"), PANEL_W // 2,
                          BRAND_H + 6, 38, (255, 255, 255), "tc", 4)
        overlay.draw_text(canvas, self.S.get("panel_robot"), PANEL_W + PANEL_W // 2,
                          BRAND_H + 6, 38, self.colors["box"], "tc", 4)
        cv2.line(canvas, (PANEL_W, PANEL_Y), (PANEL_W, PANEL_Y + PANEL_H),
                 (250, 250, 250), 3)

        self._render_bar(canvas, using_fallback)

        # Logos go on last: _render_bar repaints the whole bottom strip, and
        # anything drawn into it before that gets painted over.
        self.branding.draw(canvas, {
            "top-left": (24, BRAND_H // 2, "lm"),
            "top-right": (CANVAS_W - 24, BRAND_H // 2, "rm"),
            "bottom-left": (24, CANVAS_H - 16, "lb"),
            "bottom-right": (CANVAS_W - 24, CANVAS_H - 14, "rb"),
        })
        self.branding.draw_text_at(canvas, CANVAS_W - 24, BRAND_H // 2, 42)
        return canvas

    def _render_bar(self, canvas: np.ndarray, using_fallback: bool) -> None:
        """Bottom bar, four columns: mascot | confidence | timer | scoreboard."""
        t = self.tracker
        y0 = BAR_Y
        cv2.rectangle(canvas, (0, y0), (CANVAS_W, CANVAS_H), (26, 18, 14), -1)
        now = time.monotonic()
        frozen = now < self._frozen_until

        # --- column A: mascot + speech bubble
        if not t.visible and t.celebrating:
            state, line = "confused", self.S.get("robot_vanished")
        elif t.visible:
            state = "awake"
            line = self.S.get("robot_found") if frozen else self.S.get("robot_detected")
        else:
            state, line = "sleeping", self.S.get("robot_sleeping")
        bob = 4.0 * np.sin(now * 2.2)
        # The bar is shorter now that branding moved to the top strip, so the
        # mascot shrinks and the bubble sits clear of its antenna.
        self.mascot.draw(canvas, state, 150, CANVAS_H - 22, 172, bob)
        overlay.speech_bubble(canvas, line, 240, y0 + 96, 34, max_w=330)

        # --- column B: confidence
        bx, by, bw, bh = 440, y0 + 84, 400, 44
        overlay.confidence_bar(canvas, bx, by, bw, bh, t.conf, self.colors,
                               self.S.get("confidence"),
                               float(self.cfg["model"]["conf"]),
                               f"{self.S.get('threshold')} "
                               f"{int(float(self.cfg['model']['conf'])*100)}%")
        # Only say something when there is something to say: the camera is down,
        # or the marker board is missing. Naming the mode told the user nothing.
        line2 = ""
        if using_fallback:
            line2 = self.S.get("mode_fallback")
        elif self.mode == "virtual" and not self.board.found:
            line2 = self.S.get("board_not_found")
        if line2:
            overlay.draw_text(canvas, line2, bx, by + 84, 28,
                              (215, 215, 215), "tl", 3)
        if self.mode == "virtual":
            overlay.draw_text(canvas, poster_label(self.S, self.poster), bx,
                              by + 122, 30, self.colors["box"], "tl", 3)

        # --- column C: the big timer
        tx = 960
        overlay.draw_text(canvas, self.S.get("invisible_for"), tx, y0 + 40, 34,
                          (235, 235, 235), "tl", 4)
        if frozen:
            secs, col = self._frozen_time, self.colors["good"]
        elif t.celebrating:
            secs, col = t.invisible_seconds, self.colors["box"]
        elif t.invisible_seconds > 0:
            # Already counting, but the celebration has not fired yet: show the
            # real elapsed time dimmed, so the number never jumps by 0.6 s.
            secs, col = t.invisible_seconds, (120, 150, 165)
        else:
            secs, col = 0.0, (150, 150, 150)
        overlay.draw_text(canvas, f"{secs:.1f}{self.S.get('seconds_short')}",
                          tx, y0 + 84, 92, col, "tl", 6)
        if self._is_record and frozen:
            overlay.draw_text(canvas, self.S.get("new_record"), tx, y0 + 212, 34,
                              self.colors["good"], "tl", 5)

        # --- column D: scoreboard
        sx = 1270
        overlay.draw_text(canvas, self.S.get("top_board"), sx, y0 + 30, 30,
                          (245, 245, 245), "tl", 4)
        top = self.scores.top()
        if not top:
            overlay.draw_text(canvas, self.S.get("board_empty"), sx, y0 + 72, 28,
                              (170, 170, 170), "tl", 3)
        for i, e in enumerate(top):
            c = self.colors["good"] if i == 0 else (225, 225, 225)
            overlay.draw_text(canvas, f"{i+1}.  {e.seconds:.1f}s", sx,
                              y0 + 70 + i * 36, 30, c, "tl", 3)

        # --- footer
        overlay.draw_text(canvas, self.S.get("privacy"), 20, CANVAS_H - 10, 22,
                          (185, 185, 185), "bl", 3)
        # Right-aligned but clear of the UNICA mark in the corner.
        overlay.draw_text(canvas, self.S.get("keys_help"), CANVAS_W - 250,
                          CANVAS_H - 10, 22, (140, 140, 140), "br", 3)
        if self.cfg["ui"]["show_fps"] or self.args.show_fps:
            # Operator diagnostics: is the board seen, and is the class firing?
            bits = [f"{self._fps:4.1f} fps"]
            if self.mode == "virtual":
                bits.append("board OK" if self.board.found else "NO BOARD")
            if self.trigger == "neuron":
                fired = self._trig_score >= self._trig_conf
                bits.append(f"cls{self._trig_class} {self._trig_score:.2f} "
                            f"{'FIRE' if fired else '--'}")
            col = ((80, 230, 120) if self.trigger == "neuron"
                   and self._trig_score >= self._trig_conf else (140, 140, 140))
            overlay.draw_text(canvas, "   ".join(bits), 20, y0 + 6, 24, col, "tl", 3)

        # --- expert badge lives over the robot panel, not in the bar
        if self.expert:
            overlay.draw_text(canvas, self.S.get("expert_on"), CANVAS_W - 16,
                              PANEL_Y + 8, 30, (255, 120, 220), "tr", 5)
            overlay.draw_text(canvas, self.S.get("expert_explain"), CANVAS_W - 16,
                              PANEL_Y + 46, 24, (255, 205, 240), "tr", 4)

    def _compose_waiting(self) -> np.ndarray:
        canvas = np.full((CANVAS_H, CANVAS_W, 3), self.colors["panel_bg"], np.uint8)
        self.mascot.draw(canvas, "sleeping", CANVAS_W // 2, CANVAS_H // 2 + 60, 300)
        overlay.draw_text(canvas, self.S.get("camera_lost"), CANVAS_W // 2,
                          CANVAS_H // 2 + 110, 52, (255, 255, 255), "tc", 5)
        overlay.draw_text(canvas, self.S.get("camera_retry"), CANVAS_W // 2,
                          CANVAS_H // 2 + 180, 32, (190, 190, 190), "tc", 4)
        return canvas

    # -- input -------------------------------------------------------------
    def _handle_keys(self) -> None:
        k = cv2.waitKey(1) & 0xFF
        if k == 255:
            return
        if k == 27:
            self._running = False
        elif k in (ord("f"), ord("F")):
            self.fullscreen = not self.fullscreen
            self._apply_fullscreen()
        elif k in (ord("r"), ord("R")):
            self.scores.reset()
            log.info("scoreboard reset")
        elif k in (ord("e"), ord("E")):
            self.expert = not self.expert and self.expert_detector is not None
            self._expert_box = None
        elif k in (ord("s"), ord("S")):
            self.trigger = "detector" if self.trigger == "neuron" else "neuron"
            log.warning("trigger -> %s", self.trigger)
        elif k in (ord("v"), ord("V")):
            self._force_vanish = not self._force_vanish
            log.warning("manual force-vanish %s",
                        "ON" if self._force_vanish else "OFF")
        elif k in POSTER_KEYS:
            self.poster = POSTER_KEYS[k]
            log.info("poster -> %s", self.poster)

    def _apply_fullscreen(self) -> None:
        try:
            cv2.setWindowProperty(
                WND, cv2.WND_PROP_FULLSCREEN,
                cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        except Exception as exc:
            log.debug("fullscreen toggle failed: %s", exc)

    def shutdown(self) -> None:
        if self.camera:
            self.camera.release()
        self.audio.close()
        cv2.destroyAllWindows()


# ------------------------------------------------------------------ main ---

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def check_patch_matches_model(cfg: dict) -> None:
    """A patch only fools the model it was optimised against. Fail loudly."""
    target = str(cfg["patch"]["target_model"]).strip()
    loaded = str(cfg["model"]["name"]).strip()
    if target != loaded:
        raise ModelMismatch(
            f"\n  The patch was made for '{target}' but the demo is loading "
            f"'{loaded}'.\n"
            "  An adversarial patch only fools the model it was optimised "
            "against;\n  running this pair would make the poster do nothing at "
            "all.\n  Fix model.name / patch.target_model in config.yaml, or "
            "re-train the\n  patch with tools/train_patch.py.\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="INVISIBLE! science-fair demo")
    ap.add_argument("--lang", choices=("it", "en"), default=None)
    ap.add_argument("--mode", choices=("physical", "virtual", "fallback"),
                    default="physical",
                    help="physical (default): the user holds the real printed "
                         "poster and the app reads it. virtual: a white ArUco "
                         "board, and the operator picks the image with 1/2.")
    ap.add_argument("--camera", type=int, default=None,
                    help="camera index; never guessed, pick it explicitly")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    ap.add_argument("--poster", choices=("adversarial", "dog", "none"),
                    default=None)
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--mute", action="store_true")
    ap.add_argument("--trigger", choices=("detector", "neuron"), default=None,
                    help="detector = the box goes when YOLO really loses the "
                         "user; neuron = the class poster A was trained to "
                         "light up fires it")
    ap.add_argument("--show-board", action="store_true",
                    help="outline the detected marker board (setup aid)")
    ap.add_argument("--show-fps", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    cfg = load_config(args.config)
    if args.windowed:
        cfg["ui"]["fullscreen"] = False
    if args.poster:
        cfg["patch"]["active"] = args.poster

    try:
        check_patch_matches_model(cfg)
    except ModelMismatch as exc:
        print(f"REFUSING TO START:{exc}", file=sys.stderr)
        return 2

    try:
        show = Show(cfg, args)
    except FileNotFoundError as exc:
        print(f"REFUSING TO START: {exc}", file=sys.stderr)
        return 2
    return show.run()


if __name__ == "__main__":
    sys.exit(main())

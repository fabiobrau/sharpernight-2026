"""Headless soak test: run the real frame loop for N frames, no window.

Checks the things that matter for an unattended three-hour afternoon:
throughput, memory growth, and that no frame ever raises.

    .venv/bin/python tools/soak_test.py --frames 1200
"""
from __future__ import annotations

import argparse
import os
import resource
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as A  # noqa: E402
from compositor import make_board_image  # noqa: E402


class FakeCamera:
    """Alternates: person alone / person holding the marker board / empty room.

    Also drops frames now and then, the way a real webcam does.
    """

    def __init__(self, person_path: str) -> None:
        img = cv2.imread(person_path)
        self.person = cv2.resize(img, (1280, 720)) if img is not None else \
            np.full((720, 1280, 3), 90, np.uint8)
        self.empty = np.full((720, 1280, 3), 70, np.uint8)
        self.board = make_board_image(700)
        self.n = 0
        self.alive = True

    def read(self):
        self.n += 1
        if self.n % 97 == 0:
            return None                       # simulate a dropped frame
        phase = (self.n // 60) % 3
        if phase == 2:
            return self.empty.copy()
        frame = self.person.copy()
        if phase == 1:                        # board raised
            src = np.float32([[0, 0], [699, 0], [699, 699], [0, 699]])
            dst = np.float32([[430, 300], [810, 320], [800, 660], [440, 640]])
            M = cv2.getPerspectiveTransform(src, dst)
            cv2.warpPerspective(self.board, M, (1280, 720), dst=frame,
                                borderMode=cv2.BORDER_TRANSPARENT)
        return frame

    def release(self):
        pass


def rss_mb() -> float:
    """Peak RSS (ru_maxrss never decreases -- good for spotting a rising high-water mark)."""
    v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return v / (1024 * 1024) if sys.platform == "darwin" else v / 1024


def rss_now_mb() -> float:
    """Current RSS. Distinguishes real growth from a one-off transient spike."""
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())],
                             capture_output=True, text=True, timeout=5)
        return int(out.stdout.strip()) / 1024
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=1200)
    ap.add_argument("--expert", action="store_true")
    ap.add_argument("--person", default=os.path.join(
        ROOT, "assets/testimg/test_img_crop001024.png"))
    args = ap.parse_args()

    # Headless: swallow every window call.
    keys = iter([])
    cv2.imshow = lambda *a, **k: None
    cv2.namedWindow = lambda *a, **k: None
    cv2.setWindowProperty = lambda *a, **k: None
    cv2.destroyAllWindows = lambda *a, **k: None
    cv2.waitKey = lambda *a, **k: next(keys, 255)

    cfg = A.load_config(os.path.join(ROOT, "config.yaml"))
    cfg["ui"]["fullscreen"] = False
    cli = A.main.__wrapped__ if hasattr(A.main, "__wrapped__") else None

    import argparse as _a
    p = _a.ArgumentParser()
    for flag, kw in (("--lang", {"default": "it"}), ("--mode", {"default": "virtual"}),
                     ("--camera", {"type": int, "default": None}),
                     ("--trigger", {"default": None}),
                     ("--poster", {"default": None})):
        p.add_argument(flag, **kw)
    for flag in ("--windowed", "--mute", "--show-board", "--show-fps", "--verbose"):
        p.add_argument(flag, action="store_true")
    cli = p.parse_args(["--mute", "--windowed"])

    show = A.Show(cfg, cli)
    show.camera = FakeCamera(args.person)
    show.expert = bool(args.expert) and show.expert_detector is not None
    show.detector.warmup()

    rss0 = rss_mb()
    t0 = time.monotonic()
    errors = 0
    posters = ["adversarial", "dog", "noise", "adversarial"]
    for i in range(args.frames):
        if i % 240 == 0:                       # exercise the poster keys
            show.poster = posters[(i // 240) % len(posters)]
        try:
            show._tick()
        except Exception as exc:
            errors += 1
            print(f"  FRAME {i} RAISED: {type(exc).__name__}: {exc}")
        if i and i % 300 == 0:
            print(f"  {i:5d} frames  {i/(time.monotonic()-t0):5.1f} fps  "
                  f"rss now {rss_now_mb():6.1f} MB  peak {rss_mb():6.1f} MB  "
                  f"vanish-events {show.scores.attempts}")

    dt = time.monotonic() - t0
    rss1 = rss_mb()
    show.shutdown()

    print(f"\n  frames        : {args.frames}")
    print(f"  throughput    : {args.frames/dt:.1f} fps  ({dt:.1f}s)")
    print(f"  peak rss      : {rss0:.1f} -> {rss1:.1f} MB  (+{rss1-rss0:.1f})")
    print(f"  current rss   : {rss_now_mb():.1f} MB")
    print(f"  errors        : {errors}")
    print(f"  scored runs   : {show.scores.attempts}, best {show.scores.best:.1f}s")
    ok = errors == 0 and (rss1 - rss0) < 150
    print("  RESULT        :", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

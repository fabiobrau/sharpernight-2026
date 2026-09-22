"""Render the full screen offscreen, in every show state, to PNGs.

Lets us check the layout, the wording and both languages without opening a
fullscreen window or pointing a camera at anybody.

    .venv/bin/python tools/render_test.py --out /tmp/screens
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app as A  # noqa: E402
from compositor import make_board_image  # noqa: E402
from detector import Detection  # noqa: E402


def fake_frame(person_img: str, with_board: bool = False,
               mirror: bool = True) -> np.ndarray:
    """A stand-in camera frame, already mirrored the way Show._process would."""
    img = cv2.imread(person_img)
    if img is None:
        img = np.full((720, 1280, 3), 90, np.uint8)
    frame = cv2.resize(img, (1280, 720))
    if with_board:
        board = make_board_image(700)
        src = np.float32([[0, 0], [699, 0], [699, 699], [0, 699]])
        dst = np.float32([[430, 300], [810, 320], [800, 660], [440, 640]])
        M = cv2.getPerspectiveTransform(src, dst)
        cv2.warpPerspective(board, M, (1280, 720), dst=frame,
                            borderMode=cv2.BORDER_TRANSPARENT)
    # _process mirrors for display, so a frame built here must match or the
    # preview shows the two panels facing opposite ways.
    return cv2.flip(frame, 1) if mirror else frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/screens")
    ap.add_argument("--person", default=os.path.join(ROOT, "assets/testimg/test_img_crop001024.png"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cfg = A.load_config(os.path.join(ROOT, "config.yaml"))
    cfg["ui"]["fullscreen"] = False

    for lang in ("it", "en"):
        argv = ["--lang", lang, "--windowed", "--mute"]
        cli = _parse(argv)
        show = A.Show(cfg, cli)

        frame = fake_frame(args.person)
        clean, robot = show._process(frame)

        # 1. detected
        show.tracker.visible = True
        show.tracker.conf = 0.94
        show.tracker.box = Detection(300, 60, 760, 700, 0.94)
        show._draw_robot_view(robot)
        _save(show, clean, robot, args.out, f"{lang}_1_detected")

        # 2. vanished, timer running
        show.tracker.visible = False
        show.tracker.box = None
        show.tracker.conf = 0.06
        show.tracker._vanish_fired = True
        show.tracker._lost_since = time.monotonic() - 3.4
        robot2 = fake_frame(args.person)
        show.puff.fire(530, 380, 400)
        for _ in range(4):
            show.puff.update_and_draw(robot2)
        _save(show, clean, robot2, args.out, f"{lang}_2_vanished")

        # 3. found again, frozen time + record
        show.scores.submit(3.4)
        show.scores.submit(7.1)
        show.scores.submit(2.2)
        show.tracker.visible = True
        show.tracker.conf = 0.91
        show.tracker.box = Detection(300, 60, 760, 700, 0.91)
        show._frozen_time = 3.4
        show._frozen_until = time.monotonic() + 3
        show._is_record = False
        robot3 = fake_frame(args.person)
        show._draw_robot_view(robot3)
        _save(show, clean, robot3, args.out, f"{lang}_3_found")

        # 4. nobody there
        show.tracker.visible = False
        show.tracker.box = None
        show.tracker.conf = 0.0
        show.tracker._vanish_fired = False
        show._frozen_until = 0
        empty = np.full((720, 1280, 3), 70, np.uint8)
        _save(show, empty, empty.copy(), args.out, f"{lang}_4_sleeping")

        # 5. expert mode
        if show.expert_detector:
            show.expert = True
            show._expert_box = Detection(320, 70, 740, 690, 0.88)
            show.tracker.visible = False
            show.tracker.conf = 0.10
            show.tracker._vanish_fired = True
            show.tracker._lost_since = time.monotonic() - 1.2
            robot5 = fake_frame(args.person)
            show._draw_robot_view(robot5)
            _save(show, clean, robot5, args.out, f"{lang}_5_expert")

        # 6. camera trouble
        cv2.imwrite(os.path.join(args.out, f"{lang}_6_nocamera.png"),
                    show._compose_waiting())
        print(f"  {lang}_6_nocamera.png")
        show.shutdown()


def _save(show, clean, robot, out, name) -> None:
    canvas = show._compose(clean, robot, False)
    cv2.imwrite(os.path.join(out, name + ".png"), canvas)
    print(f"  {name}.png")


def _parse(argv):
    import argparse as _a
    p = _a.ArgumentParser()
    p.add_argument("--lang", default=None)
    p.add_argument("--mode", default="physical")
    p.add_argument("--camera", type=int, default=None)
    p.add_argument("--poster", default=None)
    p.add_argument("--windowed", action="store_true")
    p.add_argument("--mute", action="store_true")
    p.add_argument("--trigger", default=None)
    p.add_argument("--show-board", action="store_true")
    p.add_argument("--show-fps", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    main()

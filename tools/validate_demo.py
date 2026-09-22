"""End-to-end check of the real Mode B path at demo framing.

Unlike tools/validate_patch.py (which pastes pixels directly), this renders an
actual ArUco board into the frame where a child would hold it, then runs the
*shipping* code -- BoardTracker.update -> warp_into -> Detector -- and reports
how often the person ends up below threshold.

That is the number the "9 tries out of 10" acceptance criterion is about.

    .venv/bin/python tools/validate_demo.py --img-dir /path/to/photos
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compositor import BoardTracker, make_board_image, warp_into  # noqa: E402
from detector import Detector  # noqa: E402
from framing import close_framing  # noqa: E402

W, H = 1280, 720


def place_board(frame: np.ndarray, box, board: np.ndarray, height_frac: float,
                tilt: float = 0.0) -> np.ndarray:
    """Render the marker board where the child would hold it: chest, both hands."""
    x1, y1, x2, y2 = box
    bh = y2 - y1
    side = height_frac * bh
    cx = (x1 + x2) / 2
    cy = y1 + 0.42 * bh                    # chest height
    half = side / 2
    dx = tilt * half
    dst = np.float32([[cx - half + dx, cy - half], [cx + half + dx, cy - half + dx * 0.15],
                      [cx + half - dx, cy + half], [cx - half - dx, cy + half - dx * 0.15]])
    src = np.float32([[0, 0], [board.shape[1] - 1, 0],
                      [board.shape[1] - 1, board.shape[0] - 1], [0, board.shape[0] - 1]])
    out = frame.copy()
    cv2.warpPerspective(board, cv2.getPerspectiveTransform(src, dst), (W, H),
                        dst=out, borderMode=cv2.BORDER_TRANSPARENT)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--board-frac", default="0.30,0.40,0.50",
                    help="board side as a fraction of body height")
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args()

    img_dir = args.img_dir or os.environ.get("COCO_DIR", "")
    cache = os.path.join(os.path.dirname(img_dir.rstrip("/")), "person_index.json")
    if not os.path.exists(cache):
        sys.exit("run tools/train_patch.py first (it builds the person index)")
    index = json.load(open(cache))[-args.n:]      # the held-out tail

    det = Detector(os.path.join(ROOT, "assets/models/yolov8n.pt"), args.device,
                   640, args.conf, [0])
    det.warmup()
    board = make_board_image(900)
    posters = {
        "A magic": cv2.imread(os.path.join(ROOT, "assets/patches/magic_dog.png")),
        "B dog": cv2.imread(os.path.join(ROOT, "assets/decoys/dog_photo.jpg")),
        "C noise": cv2.imread(os.path.join(ROOT, "assets/decoys/noise.png")),
    }
    posters = {k: v for k, v in posters.items() if v is not None}
    fracs = [float(f) for f in args.board_frac.split(",")]

    print(f"end-to-end Mode B, threshold {args.conf}, {len(index)} held-out images")
    print(f"  {'poster':<10}" + "".join(f"  board={f:<5.2f}" for f in fracs))

    totals = {k: [] for k in posters}
    board_seen = clean_ok = trials = 0
    for frac in fracs:
        per_poster = {k: [0, 0] for k in posters}      # [vanished, trials]
        for s in index:
            img = cv2.imread(s["path"])
            if img is None:
                continue
            framed, nb = close_framing(img, s["box"])
            if framed is None:
                continue
            if not det.detect(framed):
                continue                                # must be seen when clean
            clean_ok += 1
            staged = place_board(framed, nb, board, frac)

            tracker = BoardTracker(smooth=0.0)
            quad = tracker.update(staged)
            if quad is None:
                continue                                # markers not found
            board_seen += 1
            for name, pimg in posters.items():
                composited = warp_into(staged, pimg, quad)
                gone = len(det.detect(composited)) == 0
                per_poster[name][0] += int(gone)
                per_poster[name][1] += 1
                trials += 1
                if args.save_dir and name == "A magic":
                    os.makedirs(args.save_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(
                        args.save_dir,
                        f"{frac:.2f}_{os.path.basename(s['path'])}"), composited)
        for name in posters:
            v, n = per_poster[name]
            totals[name].append((v / n * 100) if n else float("nan"))

    for name in posters:
        print(f"  {name:<10}" + "".join(f"  {p:>9.0f}%" for p in totals[name]))
    print(f"\n  board detected in {board_seen}/{clean_ok} staged frames")
    print("  (percentages = share of frames where the person fell below threshold)")


if __name__ == "__main__":
    main()

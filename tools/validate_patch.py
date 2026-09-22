"""Measure which patch actually fools which YOLO model.

Applies a patch to the torso of every detected person the way the Hu et al.
patch transformer does (square patch, side = scale * bbox diagonal, centred on
the upper torso), then re-runs detection and reports the person confidence.

Nothing here is used at demo time -- this is the evidence behind the
`patch.target_model` value recorded in config.yaml.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON = 0


def load_square(path: str, side: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.resize(img, (side, side), interpolation=cv2.INTER_CUBIC)


def noise_square(side: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    return cv2.resize(small, (side, side), interpolation=cv2.INTER_NEAREST)


def person_boxes(model, frame, conf=0.25):
    res = model.predict(frame, classes=[PERSON], conf=conf, imgsz=640,
                        device=DEVICE, verbose=False)[0]
    out = []
    for b in res.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        out.append((x1, y1, x2, y2, float(b.conf[0])))
    return out


def apply_patch(frame, box, patch_bgr, scale):
    """Paste a square patch on the upper torso, sized off the bbox diagonal."""
    x1, y1, x2, y2 = box[:4]
    bw, bh = x2 - x1, y2 - y1
    side = int(round(scale * float(np.hypot(bw, bh))))
    if side < 8:
        return frame
    cx = int(round((x1 + x2) / 2))
    cy = int(round((y1 + y2) / 2 - 0.10 * bh))
    patch = cv2.resize(patch_bgr, (side, side), interpolation=cv2.INTER_CUBIC)

    h, w = frame.shape[:2]
    px1, py1 = cx - side // 2, cy - side // 2
    sx1, sy1 = max(0, -px1), max(0, -py1)
    dx1, dy1 = max(0, px1), max(0, py1)
    dx2, dy2 = min(w, px1 + side), min(h, py1 + side)
    if dx2 <= dx1 or dy2 <= dy1:
        return frame
    out = frame.copy()
    out[dy1:dy2, dx1:dx2] = patch[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="0.20,0.25,0.30")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--models", default="yolov8n,yolov8s,yolov5nu,yolov9t,yolov10n")
    ap.add_argument("--save-dir", default="")
    args = ap.parse_args()

    global DEVICE
    DEVICE = args.device
    scales = [float(s) for s in args.scales.split(",")]

    images = sorted(glob.glob(os.path.join(ROOT, "assets/testimg/*")))
    if not images:
        sys.exit("no test images in assets/testimg/")

    candidates = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "assets/patches/*.png"))):
        candidates[os.path.splitext(os.path.basename(p))[0]] = load_square(p, 256)
    candidates["DECOY-dog"] = load_square(os.path.join(ROOT, "assets/decoys/dog_photo.jpg"), 256)
    candidates["DECOY-noise"] = noise_square(256)
    candidates["DECOY-grey"] = np.full((256, 256, 3), 128, np.uint8)

    for mname in args.models.split(","):
        model = YOLO(os.path.join(ROOT, "assets/models", mname + ".pt"))
        print(f"\n===== {mname} (device={DEVICE}) =====")

        # Clean pass: the boxes we will attack, and the baseline confidence.
        clean = {}
        for img_path in images:
            frame = cv2.imread(img_path)
            boxes = person_boxes(model, frame)
            if boxes:
                clean[img_path] = (frame, boxes)
        if not clean:
            print("  no persons detected on clean images -- skipping")
            continue
        base = np.mean([max(b[4] for b in bs) for _, bs in clean.values()])
        print(f"  clean max-person conf (mean over {len(clean)} imgs): {base:.3f}")

        hdr = "  {:<14}".format("patch") + "".join(f"  s={s:<6.2f}" for s in scales)
        print(hdr)
        rows = []
        for cname, cimg in candidates.items():
            cells = []
            for scale in scales:
                confs = []
                for img_path, (frame, boxes) in clean.items():
                    patched = frame
                    for b in boxes:
                        patched = apply_patch(patched, b, cimg, scale)
                    got = person_boxes(model, patched)
                    confs.append(max((g[4] for g in got), default=0.0))
                    if args.save_dir and scale == scales[0]:
                        os.makedirs(args.save_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(
                            args.save_dir,
                            f"{mname}_{cname}_{os.path.basename(img_path)}.jpg"), patched)
                cells.append(float(np.mean(confs)))
            rows.append((cname, cells))
            print("  {:<14}".format(cname) + "".join(f"  {c:<8.3f}" for c in cells))

        best = min(rows, key=lambda r: min(r[1]))
        print(f"  --> strongest: {best[0]} (min mean conf {min(best[1]):.3f})")


if __name__ == "__main__":
    main()

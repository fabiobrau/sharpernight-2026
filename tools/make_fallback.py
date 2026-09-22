"""Build Mode C content: genuine before/after stills, and a looping clip.

These are not mock-ups. Each pair is produced by actually running the detector
twice -- once on the clean photo, once with the real patch composited in -- so
what the queue sees while the camera is busy is the same effect the camera
would have shown them.

    .venv/bin/python tools/make_fallback.py --img-dir /path/to/photos
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

import overlay  # noqa: E402
from strings import Strings  # noqa: E402

W, H = 1280, 720
BOX = (60, 220, 255)
GOOD = (80, 230, 120)


def paste(frame, box, patch, scale=0.32):
    x1, y1, x2, y2 = box
    side = int(round(scale * float(np.hypot(x2 - x1, y2 - y1))))
    if side < 8:
        return frame
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2 - 0.10 * (y2 - y1))
    p = cv2.resize(patch, (side, side), interpolation=cv2.INTER_CUBIC)
    out = frame.copy()
    h, w = out.shape[:2]
    dx1, dy1 = max(0, cx - side // 2), max(0, cy - side // 2)
    dx2, dy2 = min(w, dx1 + side), min(h, dy1 + side)
    out[dy1:dy2, dx1:dx2] = p[:dy2 - dy1, :dx2 - dx1]
    return out


def panel(img, title, box, conf, S, ok_color):
    img = cv2.resize(img, (W // 2, H - 90))
    if box is not None:
        sx, sy = (W // 2) / W, (H - 90) / H
        overlay.corner_box(img, (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy),
                           BOX, 8)
        overlay.draw_text(img, f"{S.get('label_human')} {int(conf*100)}%",
                          box[0] * sx, box[1] * sy - 10, 32, BOX, "bl", 4)
    else:
        overlay.draw_text(img, S.get("robot_vanished"), img.shape[1] // 2,
                          img.shape[0] // 2, 46, ok_color, "cm", 5)
    return img


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--lang", default="it")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    from ultralytics import YOLO
    S = Strings(args.lang)
    overlay.set_font(os.path.join(ROOT, "assets/fonts/Fredoka.ttf"))
    y = YOLO(os.path.join(ROOT, "assets/models/yolov8n.pt"))
    patch = cv2.imread(os.path.join(ROOT, "assets/patches/magic_dog.png"))
    if patch is None:
        sys.exit("train the patch first: tools/train_patch.py")

    img_dir = args.img_dir or os.environ.get("COCO_DIR", "")
    cache = os.path.join(os.path.dirname(img_dir.rstrip("/")), "person_index.json")
    if os.path.exists(cache):
        index = json.load(open(cache))
    elif img_dir and os.path.isdir(img_dir):
        index = [{"path": os.path.join(img_dir, f)} for f in sorted(os.listdir(img_dir))]
    else:
        sys.exit("pass --img-dir with photos of people")

    out_dir = os.path.join(ROOT, "assets/fallback/stills")
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.endswith(".jpg"):
            os.remove(os.path.join(out_dir, f))

    def detect(im):
        r = y.predict(im, classes=[0], conf=0.4, imgsz=640, device=args.device,
                      verbose=False)[0]
        bs = [(*b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]
        return max(bs, key=lambda b: (b[2]-b[0])*(b[3]-b[1])) if bs else None

    made = 0
    for s in index:
        if made >= args.n:
            break
        img = cv2.imread(s["path"])
        if img is None:
            continue
        img = cv2.resize(img, (W, H))
        clean_box = detect(img)
        if clean_box is None:
            continue
        patched = paste(img, clean_box[:4], patch)
        after_box = detect(patched)
        if after_box is not None:
            continue                      # only keep pairs where it truly works

        canvas = np.full((H, W, 3), (40, 28, 22), np.uint8)
        canvas[60:H - 30, 0:W // 2] = panel(img, "", clean_box[:4], clean_box[4], S, GOOD)
        canvas[60:H - 30, W // 2:W] = panel(patched, "", None, 0, S, GOOD)
        overlay.draw_text(canvas, S.get("panel_robot"), W // 4, 10, 32,
                          (255, 255, 255), "tc", 4)
        overlay.draw_text(canvas, S.get("poster_magic"), W * 3 // 4, 10, 32,
                          BOX, "tc", 4)
        overlay.draw_text(canvas, S.get("privacy"), 16, H - 6, 20,
                          (185, 185, 185), "bl", 3)
        cv2.imwrite(os.path.join(out_dir, f"still_{made:02d}.jpg"), canvas,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        made += 1
        print(f"  still_{made-1:02d}.jpg  (clean {clean_box[4]:.2f} -> undetected)")

    if made == 0:
        sys.exit("no image produced a clean before/after pair")

    # A gentle loop so the screen is never static while the queue waits.
    clip = os.path.join(ROOT, "assets/fallback/loop.mp4")
    vw = cv2.VideoWriter(clip, cv2.VideoWriter_fourcc(*"mp4v"), 25, (W, H))
    for i in range(made):
        f = cv2.imread(os.path.join(out_dir, f"still_{i:02d}.jpg"))
        for _ in range(75):               # 3 s each
            vw.write(f)
    vw.release()
    print(f"  loop.mp4 ({made * 3}s, {os.path.getsize(clip)//1024} KB)")


if __name__ == "__main__":
    main()

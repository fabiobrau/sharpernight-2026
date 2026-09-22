"""Optimise a naturalistic adversarial patch against the model we actually ship.

Why this exists: the published patches (Hu et al. / Bimo99B9) measure as plain
occlusion on the ultralytics weights we deploy -- a dog photo, random noise and
a grey square all suppress person confidence by the same amount. A demo built on
that is a demo where the decoy posters work as well as the "magic" one, which
destroys the point. So we train our own against these exact weights.

It stays a *naturalistic* patch: an anchor loss keeps it looking like the fluffy
dog it started from, and TV + non-printability losses keep it printable, so the
same file works for the virtual board (Mode B) and the printed poster (Mode A).

    .venv/bin/python tools/train_patch.py --steps 1500
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from framing import close_framing  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERSON = 0

# What the app actually feeds the detector. The camera frame is 1280x720 and
# ultralytics letterboxes it to 640x384 -- i.e. HALF scale. Training on square
# 640x640 crops would render the patch at twice its deployed pixel size, and an
# adversarial pattern does not survive that (see docs/PATCH_NOTES.md).
CAM_W, CAM_H = 1280, 720
NET_W, NET_H = 640, 384
_SCALE = NET_W / CAM_W                       # 0.5
_PAD_Y = (NET_H - int(CAM_H * _SCALE)) // 2  # 12 rows top and bottom
CANVAS = NET_W                               # patch-scale reference

# Printable colour triplets (30 swatches, RGB 0-1) -- standard NPS palette.
PRINTABLE = np.array([
    [0.10, 0.10, 0.10], [0.25, 0.25, 0.25], [0.45, 0.45, 0.45], [0.65, 0.65, 0.65],
    [0.90, 0.90, 0.90], [0.55, 0.11, 0.13], [0.78, 0.20, 0.18], [0.90, 0.36, 0.30],
    [0.40, 0.09, 0.12], [0.93, 0.62, 0.55], [0.13, 0.33, 0.16], [0.21, 0.51, 0.24],
    [0.40, 0.70, 0.35], [0.62, 0.83, 0.50], [0.09, 0.22, 0.12], [0.11, 0.20, 0.45],
    [0.16, 0.33, 0.64], [0.28, 0.51, 0.80], [0.55, 0.72, 0.90], [0.07, 0.13, 0.30],
    [0.85, 0.72, 0.13], [0.95, 0.85, 0.30], [0.99, 0.94, 0.62], [0.60, 0.48, 0.09],
    [0.45, 0.20, 0.55], [0.63, 0.36, 0.71], [0.80, 0.60, 0.85], [0.85, 0.45, 0.10],
    [0.95, 0.65, 0.25], [0.35, 0.22, 0.12],
], dtype=np.float32)


# ----------------------------------------------------------------- data ----

def build_index(img_dir: str, cache: str, model, device: str,
                want: int = 500) -> list[dict]:
    """Find COCO images with one clear, reasonably large person; cache the boxes."""
    if os.path.exists(cache):
        with open(cache) as f:
            idx = json.load(f)
        if len(idx) >= want:
            print(f"index: {len(idx)} cached samples")
            return idx[:want]

    files = sorted(os.listdir(img_dir))
    random.Random(0).shuffle(files)
    idx: list[dict] = []
    t0 = time.time()
    for n, fn in enumerate(files):
        if len(idx) >= want:
            break
        p = os.path.join(img_dir, fn)
        img = cv2.imread(p)
        if img is None:
            continue
        res = model.predict(img, classes=[PERSON], conf=0.60, imgsz=CANVAS,
                            device=device, verbose=False)[0]
        boxes = [b.xyxy[0].tolist() for b in res.boxes]
        if not boxes:
            continue
        box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        ih, iw = img.shape[:2]
        w, h = box[2] - box[0], box[3] - box[1]
        # Boxes are kept in ORIGINAL image coordinates: close_framing re-crops
        # from the full-resolution photo, so a chest-sized poster still has real
        # pixels behind it.
        if w * h < 0.03 * iw * ih or h < 0.25 * ih:
            continue
        idx.append({"path": p, "box": box})
        if n % 400 == 0:
            print(f"  scanned {n}, kept {len(idx)} ({time.time()-t0:.0f}s)")
    with open(cache, "w") as f:
        json.dump(idx, f)
    print(f"index: {len(idx)} samples ({time.time()-t0:.0f}s)")
    return idx


def letterbox(frame: np.ndarray) -> np.ndarray:
    """Reproduce ultralytics' preprocessing for a 1280x720 frame exactly."""
    small = cv2.resize(frame, (NET_W, int(CAM_H * _SCALE)),
                       interpolation=cv2.INTER_LINEAR)
    out = np.full((NET_H, NET_W, 3), 114, np.uint8)
    out[_PAD_Y:_PAD_Y + small.shape[0]] = small
    return out


def load_batch(samples: list[dict], device: str, rng: random.Random | None = None,
               close: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch exactly as the running demo will present it to the detector.

    close_framing puts the person at demo distance in a 1280x720 camera frame;
    letterbox() then applies the same half-scale resize ultralytics does. `fill`
    is randomised so the patch works across the distances a child may stand at.
    """
    imgs, boxes = [], []
    for s in samples:
        img = cv2.imread(s["path"])
        if img is None:
            continue
        if close:
            fill = rng.uniform(0.55, 0.92) if rng else 0.78
            framed, box = close_framing(img, s["box"], fill,
                                        out_w=CAM_W, out_h=CAM_H)
            if framed is None:
                continue
        else:
            framed = cv2.resize(img, (CAM_W, CAM_H))
            h, w = img.shape[:2]
            sx, sy = CAM_W / w, CAM_H / h
            box = [s["box"][0] * sx, s["box"][1] * sy,
                   s["box"][2] * sx, s["box"][3] * sy]

        imgs.append(torch.from_numpy(letterbox(framed)[:, :, ::-1].copy())
                    .permute(2, 0, 1).float() / 255)
        boxes.append([box[0] * _SCALE, box[1] * _SCALE + _PAD_Y,
                      box[2] * _SCALE, box[3] * _SCALE + _PAD_Y])
    if not imgs:
        return None, None
    return (torch.stack(imgs).to(device),
            torch.tensor(boxes, dtype=torch.float32, device=device))


# ------------------------------------------------------------------ EOT ----

SCALE_RANGE = (0.22, 0.42)      # patch side as a fraction of the bbox diagonal


def paste(x: torch.Tensor, patch: torch.Tensor, boxes: torch.Tensor,
          rng: random.Random, train: bool = True) -> torch.Tensor:
    """Differentiably paste the patch onto each person's torso, with EOT jitter."""
    B, _, H, W = x.shape
    a, b = W / 2.0, H / 2.0
    thetas = []
    for i in range(B):
        x1, y1, x2, y2 = boxes[i].tolist()
        bw, bh = x2 - x1, y2 - y1
        diag = math.hypot(bw, bh)

        if train:
            scale = rng.uniform(*SCALE_RANGE)
            ang = math.radians(rng.uniform(-22, 22))
            jx = rng.uniform(-0.09, 0.09) * bw
            jy = rng.uniform(-0.08, 0.08) * bh
        else:
            scale, ang, jx, jy = 0.30, 0.0, 0.0, 0.0

        s = max(8.0, scale * diag)
        cx = (x1 + x2) / 2 + jx
        cy = (y1 + y2) / 2 - 0.10 * bh + jy

        cos, sin = math.cos(-ang), math.sin(-ang)
        R = torch.tensor([[cos, sin], [-sin, cos]], device=x.device)
        A = torch.tensor([[a, 0.0, a - cx], [0.0, b, b - cy]], device=x.device)
        thetas.append((2.0 / s) * (R @ A))
    theta = torch.stack(thetas)

    p = patch.unsqueeze(0).expand(B, -1, -1, -1)
    if train:
        # Printing, lighting and camera noise the poster will actually meet.
        contrast = torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.80, 1.20)
        bright = torch.empty(B, 1, 1, 1, device=x.device).uniform_(-0.12, 0.12)
        p = (p * contrast + bright).clamp(0, 1)

    grid = F.affine_grid(theta, (B, 3, H, W), align_corners=False)
    warped = F.grid_sample(p, grid, align_corners=False, padding_mode="zeros")
    ones = torch.ones(B, 1, patch.shape[1], patch.shape[2], device=x.device)
    mask = F.grid_sample(ones, grid, align_corners=False, padding_mode="zeros")

    if train:
        warped = warped + torch.randn_like(warped) * rng.uniform(0.0, 0.03)
        warped = warped.clamp(0, 1)
    return x * (1 - mask) + warped * mask


# --------------------------------------------------------------- losses ----

def tv_loss(p: torch.Tensor) -> torch.Tensor:
    dh = (p[:, 1:, :] - p[:, :-1, :]).abs().mean()
    dw = (p[:, :, 1:] - p[:, :, :-1]).abs().mean()
    return dh + dw


def nps_loss(p: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    d = (p.permute(1, 2, 0).unsqueeze(-2) - palette).pow(2).sum(-1).sqrt()
    return d.min(dim=-1).values.mean()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--weights", default="assets/models/yolov8n.pt")
    ap.add_argument("--anchor", default="assets/patches/exp53.png")
    ap.add_argument("--w-anchor", type=float, default=2.5)
    ap.add_argument("--w-tv", type=float, default=0.30)
    ap.add_argument("--w-nps", type=float, default=0.06)
    ap.add_argument("--out", default="assets/patches/magic_dog.png")
    ap.add_argument("--img-dir", default="")
    ap.add_argument("--objective", choices=("evade", "neuron"), default="evade",
                    help="evade = suppress the person score (a real attack, hard); "
                         "neuron = drive one output class up (easy, and all a "
                         "trigger needs)")
    ap.add_argument("--scale-min", type=float, default=0.22,
                    help="smallest patch size seen in training, as a fraction of "
                         "the body diagonal. The marker board eats a third of the "
                         "poster, so the patch lands smaller than you expect.")
    ap.add_argument("--scale-max", type=float, default=0.42)
    ap.add_argument("--target-class", type=int, default=78,
                    help="class to light up for --objective neuron (78 = hair drier, "
                         "which will never be in frame at a science fair)")
    ap.add_argument("--wide-framing", action="store_true",
                    help="train on whole photos instead of demo-like close framing")
    args = ap.parse_args()

    global SCALE_RANGE
    SCALE_RANGE = (args.scale_min, args.scale_max)
    dev = args.device
    torch.manual_seed(1234)
    rng = random.Random(1234)

    from ultralytics import YOLO
    y = YOLO(os.path.join(ROOT, args.weights))
    net = y.model.to(dev).eval()
    for q in net.parameters():
        q.requires_grad_(False)

    img_dir = args.img_dir or os.environ.get("COCO_DIR", "")
    if not img_dir or not os.path.isdir(img_dir):
        raise SystemExit("pass --img-dir pointing at a folder of photos of people")
    cache = os.path.join(os.path.dirname(img_dir.rstrip("/")), "person_index.json")
    index = build_index(img_dir, cache, y, dev, args.images)
    if len(index) < 32:
        raise SystemExit("not enough person images found")
    split = int(len(index) * 0.85)
    train_idx, val_idx = index[:split], index[split:]

    anchor_bgr = cv2.imread(os.path.join(ROOT, args.anchor))
    if anchor_bgr is None:
        raise SystemExit(f"anchor image not found: {args.anchor}")
    anchor_bgr = cv2.resize(anchor_bgr, (args.size, args.size), interpolation=cv2.INTER_CUBIC)
    anchor = torch.from_numpy(anchor_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(dev)

    # Start from the dog itself: keeps the look and gives the optimiser a head start.
    raw = torch.logit(anchor.clamp(0.02, 0.98)).clone().requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    palette = torch.from_numpy(PRINTABLE).to(dev)

    print(f"training {args.size}px patch vs {os.path.basename(args.weights)} "
          f"on {len(train_idx)} images, {args.steps} steps")
    t0 = time.time()
    best = float("inf")
    ema = None                     # single-batch adv is noisy; checkpoint on EMA
    for step in range(1, args.steps + 1):
        batch = [rng.choice(train_idx) for _ in range(args.batch)]
        x, boxes = load_batch(batch, dev, rng, close=not args.wide_framing)
        if x is None:
            continue
        patch = torch.sigmoid(raw)

        xp = paste(x, patch, boxes, rng, train=True)
        pred = net(xp)[0]                      # (B, 4 + nc, anchors)
        if args.objective == "neuron":
            # Just make one output neuron shout. No need to beat the detector at
            # its own job -- a trigger only needs a signal it can read.
            tgt = pred[:, 4 + args.target_class, :]
            # top-5 rather than top-20: we want a few anchors shouting clearly,
            # not a faint glow spread over the whole grid.
            adv = -(2.0 * tgt.max(dim=1).values.mean()
                    + tgt.topk(5, dim=1).values.mean())
        else:
            person = pred[:, 4, :]
            # max drives it down hard; top-k mean keeps the gradient informative
            adv = (person.max(dim=1).values.mean()
                   + person.topk(20, dim=1).values.mean())

        l_anchor = F.mse_loss(patch, anchor)
        l_tv = tv_loss(patch)
        l_nps = nps_loss(patch, palette)
        loss = adv + args.w_anchor * l_anchor + args.w_tv * l_tv + args.w_nps * l_nps

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if step % 50 == 0 or step == 1:
            print(f"  step {step:4d}  adv {adv.item():.4f}  anchor {l_anchor.item():.4f}"
                  f"  tv {l_tv.item():.4f}  nps {l_nps.item():.4f}"
                  f"  ({time.time()-t0:.0f}s)")
        ema = adv.item() if ema is None else 0.95 * ema + 0.05 * adv.item()
        if step % 100 == 0 and ema < best:
            best = ema
            _save(patch, os.path.join(ROOT, args.out))

    patch = torch.sigmoid(raw).detach()
    _save(patch, os.path.join(ROOT, args.out))
    print(f"saved {args.out} ({time.time()-t0:.0f}s)")

    # Honest check on images the patch never trained on, through the real API.
    print("\nheld-out check (ultralytics inference path):")
    if args.objective == "neuron":
        _evaluate_neuron(y, val_idx, os.path.join(ROOT, args.out), dev,
                         args.target_class)
    else:
        _evaluate(y, val_idx, os.path.join(ROOT, args.out), dev)


def _evaluate_neuron(y, val_idx, patch_path, device, target_class) -> None:
    """How reliably does the target class fire for the patch, and never for the decoy?"""
    name = y.names[target_class]
    cands = {"A magic (trained)": cv2.imread(patch_path),
             "B decoy dog": cv2.imread(os.path.join(ROOT, "assets/decoys/dog_photo.jpg")),
             "nothing held": None}
    rows = {}
    for label, pimg in cands.items():
        confs = []
        for s in val_idx:
            img = cv2.imread(s["path"])
            if img is None:
                continue
            framed, b = close_framing(img, s["box"], 0.78, out_w=CAM_W, out_h=CAM_H)
            if framed is None:
                continue
            if pimg is not None:
                x = torch.from_numpy(letterbox(framed)[:, :, ::-1].copy())
                x = x.permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
                box = torch.tensor([[b[0] * _SCALE, b[1] * _SCALE + _PAD_Y,
                                     b[2] * _SCALE, b[3] * _SCALE + _PAD_Y]],
                                   dtype=torch.float32, device=device)
                t = cv2.resize(pimg, (256, 256), interpolation=cv2.INTER_CUBIC)
                t = torch.from_numpy(t[:, :, ::-1].copy()).permute(2, 0, 1)
                t = t.float().div(255).to(device)
                out = paste(x, t, box, random.Random(0), train=False)
                arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                framed = np.ascontiguousarray(arr[_PAD_Y:NET_H - _PAD_Y][:, :, ::-1])
            r = y.predict(framed, classes=[target_class], conf=0.10, imgsz=640,
                          device=device, verbose=False)[0]
            confs.append(max([float(bx.conf[0]) for bx in r.boxes], default=0.0))
        rows[label] = confs
    print(f"  target class {target_class} = '{name}', {len(next(iter(rows.values())))} held-out images")
    print(f"  {'held':<20}{'mean conf':>10}{'fires >0.5':>12}{'fires >0.25':>13}")
    for label, confs in rows.items():
        a = float(np.mean([c > 0.50 for c in confs])) * 100
        b2 = float(np.mean([c > 0.25 for c in confs])) * 100
        print(f"  {label:<20}{np.mean(confs):>10.3f}{a:>11.0f}%{b2:>12.0f}%")


def _save(patch: torch.Tensor, path: str) -> None:
    arr = (patch.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(path, arr[:, :, ::-1])


def _evaluate(y, val_idx, patch_path, device) -> None:
    """Compare the trained patch against the two decoys on unseen images."""
    import glob

    cands = {"magic (trained)": cv2.imread(patch_path)}
    for name, path in (("decoy dog", "assets/decoys/dog_photo.jpg"),
                       ("decoy noise", "assets/decoys/noise.png")):
        im = cv2.imread(os.path.join(ROOT, path))
        if im is not None:
            cands[name] = im
    cands["grey square"] = np.full((256, 256, 3), 128, np.uint8)

    def run(img):
        r = y.predict(img, classes=[PERSON], conf=0.25, imgsz=640,
                      device=device, verbose=False)[0]
        return max([float(b.conf[0]) for b in r.boxes], default=0.0)

    clean, results = [], {k: [] for k in cands}
    for s in val_idx:
        img = cv2.imread(s["path"])
        if img is None:
            continue
        img, b = close_framing(img, s["box"], 0.78, out_w=CAM_W, out_h=CAM_H)
        if img is None:
            continue
        clean.append(run(img))
        lb = letterbox(img)
        x = torch.from_numpy(lb[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)
        x = x.unsqueeze(0).to(device)
        box = torch.tensor([[b[0] * _SCALE, b[1] * _SCALE + _PAD_Y,
                             b[2] * _SCALE, b[3] * _SCALE + _PAD_Y]],
                           dtype=torch.float32, device=device)
        for name, cimg in cands.items():
            c = cv2.resize(cimg, (256, 256), interpolation=cv2.INTER_CUBIC)
            t = torch.from_numpy(c[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).to(device)
            out = paste(x, t, box, random.Random(0), train=False)
            arr = (out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            # strip the letterbox padding so ultralytics re-letterboxes cleanly
            arr = arr[_PAD_Y:NET_H - _PAD_Y]
            results[name].append(run(np.ascontiguousarray(arr)))

    n = len(clean)
    print(f"  {n} held-out images; clean mean person conf {np.mean(clean):.3f}")
    print(f"  {'poster':<18}{'mean conf':>10}{'undetected':>12}")
    for name, vals in results.items():
        gone = float(np.mean([v < 0.4 for v in vals])) * 100
        print(f"  {name:<18}{np.mean(vals):>10.3f}{gone:>11.0f}%")


if __name__ == "__main__":
    main()

"""Shared framing helper.

The demo camera sees one child, close, filling most of the frame. COCO sees
people at every distance, mostly small. Training on COCO framing and then
deploying at demo framing was measured to lose almost all of the attack -- the
patch is rendered at a very different pixel resolution in the detector's 640px
input, and it does not transfer.

So both the trainer and the validator re-crop through this function.
"""
from __future__ import annotations

import cv2
import numpy as np

W, H = 1280, 720


def close_framing(img: np.ndarray, box: list[float], fill: float = 0.78,
                  out_w: int = W, out_h: int = H, max_pad: float = 0.35):
    """Re-crop so the person fills `fill` of the frame height.

    Returns (framed_image, box_in_new_coords) or (None, None).
    Pads by replication rather than clamping, so the person stays centred and
    the requested scale is exact -- but a crop that would be mostly smeared
    replication is rejected instead, because training on those teaches the
    patch about padding artefacts rather than about people.
    """
    x1, y1, x2, y2 = box
    bh = max(1.0, y2 - y1)
    target_h = bh / max(1e-3, fill)
    target_w = target_h * (out_w / out_h)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    h, w = img.shape[:2]
    cx1, cy1 = cx - target_w / 2, cy - target_h / 2
    cx2, cy2 = cx + target_w / 2, cy + target_h / 2

    vis_w = max(0.0, min(cx2, w) - max(cx1, 0.0))
    vis_h = max(0.0, min(cy2, h) - max(cy1, 0.0))
    if vis_w * vis_h < (1.0 - max_pad) * target_w * target_h:
        return None, None

    pl, pt = int(max(0, -cx1)) + 1, int(max(0, -cy1)) + 1
    pr, pb = int(max(0, cx2 - w)) + 1, int(max(0, cy2 - h)) + 1
    img = cv2.copyMakeBorder(img, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    cx1, cy1, cx2, cy2 = cx1 + pl, cy1 + pt, cx2 + pl, cy2 + pt

    crop = img[int(cy1):int(cy2), int(cx1):int(cx2)]
    if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None, None
    sx, sy = out_w / crop.shape[1], out_h / crop.shape[0]
    out = cv2.resize(crop, (out_w, out_h))
    nb = [(x1 - cx1) * sx, (y1 - cy1) * sy, (x2 - cx1) * sx, (y2 - cy1) * sy]
    return out, nb

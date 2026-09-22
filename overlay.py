"""All the cartoon drawing: fat boxes, health bar, particles, mascot, text.

Text goes through Pillow (OpenCV's Hershey fonts are far too thin to read from
3 m) but every string is rendered once and cached as an RGBA sprite, so the
per-frame cost is a numpy alpha blit, not a PIL round-trip.
"""
from __future__ import annotations

import logging
import math
import os
import random
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("invisible.overlay")

_FONT_PATH = "assets/fonts/Fredoka.ttf"


def set_font(path: str) -> None:
    global _FONT_PATH
    if os.path.exists(path):
        _FONT_PATH = path
    else:
        log.warning("font %s missing; falling back to Pillow default", path)


@lru_cache(maxsize=32)
def _font(size: int, weight: int = 600):
    try:
        f = ImageFont.truetype(_FONT_PATH, size)
        try:
            f.set_variation_by_axes([float(weight), 100.0])
        except Exception:
            pass  # static font: nothing to vary
        return f
    except Exception as exc:
        log.warning("font load failed (%s); using default", exc)
        return ImageFont.load_default()


@lru_cache(maxsize=512)
def _text_sprite(text: str, size: int, color: tuple, outline: int,
                 outline_color: tuple, weight: int) -> np.ndarray:
    """Render text once into a BGRA sprite. Cached: the timer only ticks 10x/s."""
    font = _font(size, weight)
    pad = outline + max(2, size // 8)
    dummy = Image.new("RGBA", (1, 1))
    box = ImageDraw.Draw(dummy).textbbox((0, 0), text, font=font,
                                         stroke_width=outline)
    w = max(1, box[2] - box[0] + 2 * pad)
    h = max(1, box[3] - box[1] + 2 * pad)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ImageDraw.Draw(img).text(
        (pad - box[0], pad - box[1]), text, font=font,
        fill=(color[2], color[1], color[0], 255), stroke_width=outline,
        stroke_fill=(outline_color[2], outline_color[1], outline_color[0], 255))
    rgba = np.array(img)
    bgra = rgba[:, :, [2, 1, 0, 3]].copy()
    return bgra


def alpha_blit(dst: np.ndarray, sprite: np.ndarray, x: int, y: int,
               anchor: str = "tl", scale: float = 1.0) -> tuple[int, int]:
    """Blit a BGRA sprite onto a BGR frame. Returns the drawn (w, h)."""
    if sprite is None or sprite.size == 0:
        return 0, 0
    if scale != 1.0:
        sh, sw = sprite.shape[:2]
        nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
        sprite = cv2.resize(sprite, (nw, nh), interpolation=cv2.INTER_AREA)
    h, w = sprite.shape[:2]

    if "c" in anchor:
        x -= w // 2
    if "r" in anchor:
        x -= w
    if "m" in anchor:
        y -= h // 2
    if "b" in anchor:
        y -= h

    H, W = dst.shape[:2]
    sx1, sy1 = max(0, -x), max(0, -y)
    dx1, dy1 = max(0, x), max(0, y)
    dx2, dy2 = min(W, x + w), min(H, y + h)
    if dx2 <= dx1 or dy2 <= dy1:
        return w, h

    src = sprite[sy1:sy1 + (dy2 - dy1), sx1:sx1 + (dx2 - dx1)]
    a = src[:, :, 3:4].astype(np.float32) / 255.0
    region = dst[dy1:dy2, dx1:dx2].astype(np.float32)
    dst[dy1:dy2, dx1:dx2] = (region * (1 - a) + src[:, :, :3].astype(np.float32) * a
                             ).astype(np.uint8)
    return w, h


def draw_text(dst, text, x, y, size=48, color=(255, 255, 255), anchor="tl",
              outline=4, outline_color=(30, 20, 15), weight=600):
    sprite = _text_sprite(str(text), int(size), tuple(color), int(outline),
                          tuple(outline_color), int(weight))
    return alpha_blit(dst, sprite, int(x), int(y), anchor)


def text_size(text, size=48, outline=4, weight=600) -> tuple[int, int]:
    s = _text_sprite(str(text), int(size), (255, 255, 255), int(outline),
                     (0, 0, 0), int(weight))
    return s.shape[1], s.shape[0]


# --------------------------------------------------------------------------
# shapes
# --------------------------------------------------------------------------

def rounded_rect(img, p1, p2, color, thickness=10, radius=28):
    x1, y1 = int(p1[0]), int(p1[1])
    x2, y2 = int(p2[0]), int(p2[1])
    r = int(max(0, min(radius, abs(x2 - x1) // 2, abs(y2 - y1) // 2)))
    if thickness < 0:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    else:
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, cv2.LINE_AA)
    for cx, cy, a in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                      (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), a, 0, 90, color,
                    thickness, cv2.LINE_AA)


def corner_box(img, box, color, thickness=12, frac=0.28, radius=24):
    """The detection box: fat cartoon corner brackets, not a thin rectangle."""
    x1, y1, x2, y2 = (int(v) for v in box)
    w, h = x2 - x1, y2 - y1
    lx, ly = int(w * frac), int(h * frac)
    r = min(radius, lx, ly)
    segs = [
        ((x1 + r, y1), (x1 + lx, y1)), ((x1, y1 + r), (x1, y1 + ly)),
        ((x2 - lx, y1), (x2 - r, y1)), ((x2, y1 + r), (x2, y1 + ly)),
        ((x1 + r, y2), (x1 + lx, y2)), ((x1, y2 - ly), (x1, y2 - r)),
        ((x2 - lx, y2), (x2 - r, y2)), ((x2, y2 - ly), (x2, y2 - r)),
    ]
    for a, b in segs:
        cv2.line(img, a, b, color, thickness, cv2.LINE_AA)
    for cx, cy, ang in ((x1 + r, y1 + r, 180), (x2 - r, y1 + r, 270),
                        (x2 - r, y2 - r, 0), (x1 + r, y2 - r, 90)):
        cv2.ellipse(img, (cx, cy), (r, r), ang, 0, 90, color,
                    thickness, cv2.LINE_AA)


def confidence_bar(img, x, y, w, h, value, colors, label="", threshold=None,
                   thr_label=""):
    """Video-game health bar. Drains as the detector loses the child."""
    value = float(np.clip(value, 0.0, 1.0))
    rounded_rect(img, (x, y), (x + w, y + h), (28, 20, 16), -1, h // 2)
    fill_w = int(w * value)
    if fill_w > h // 2:
        if value > 0.66:
            col = colors["good"]
        elif value > 0.33:
            col = (60, 200, 250)
        else:
            col = colors["bad"]
        rounded_rect(img, (x + 3, y + 3), (x + fill_w - 3, y + h - 3), col, -1,
                     (h - 6) // 2)
    rounded_rect(img, (x, y), (x + w, y + h), (240, 240, 240), 4, h // 2)

    if threshold is not None:
        tx = int(x + w * float(threshold))
        cv2.line(img, (tx, y - 6), (tx, y + h + 6), (250, 250, 250), 3, cv2.LINE_AA)
        if thr_label:
            draw_text(img, thr_label, tx, y + h + 10, 22, (235, 235, 235), "tc", 3)
    if label:
        draw_text(img, label, x, y - 12, 30, (255, 255, 255), "bl", 4)
    draw_text(img, f"{int(round(value * 100))}%", x + w + 18, y + h // 2, 40,
              (255, 255, 255), "lm", 4)


# --------------------------------------------------------------------------
# particles
# --------------------------------------------------------------------------

class PuffBurst:
    """Puff-of-smoke particles for the moment the box pops."""

    def __init__(self) -> None:
        self._p: list[list[float]] = []

    def fire(self, cx: float, cy: float, size: float, n: int = 40) -> None:
        for _ in range(n):
            ang = random.uniform(0, math.tau)
            spd = random.uniform(0.3, 0.9) * size * 0.020
            # start already spread around the box, so frame 1 reads as a puff
            r0 = random.uniform(0.0, 0.30) * size
            self._p.append([cx + math.cos(ang) * r0, cy + math.sin(ang) * r0,
                            math.cos(ang) * spd, math.sin(ang) * spd - 0.4,
                            random.uniform(size * 0.035, size * 0.085),
                            1.0, random.uniform(0.022, 0.040)])

    @property
    def alive(self) -> bool:
        return bool(self._p)

    def update_and_draw(self, img, color=(245, 245, 245)) -> None:
        """One overlay for the whole burst -- never copy the frame per particle."""
        keep = []
        for p in self._p:
            p[0] += p[2]; p[1] += p[3]
            p[3] += 0.05                      # gentle gravity
            p[2] *= 0.94; p[3] *= 0.94
            p[4] *= 1.012                     # puffs expand as they fade
            p[5] -= p[6]
            if p[5] > 0:
                keep.append(p)
        self._p = keep
        if not keep:
            return

        ov = img.copy()
        strongest = 0.0
        for p in keep:
            r = int(p[4])
            if r < 1:
                continue
            cv2.circle(ov, (int(p[0]), int(p[1])), r, color, -1, cv2.LINE_AA)
            strongest = max(strongest, p[5])
        a = float(np.clip(strongest, 0.0, 1.0)) * 0.7
        cv2.addWeighted(ov, a, img, 1 - a, 0, img)


# --------------------------------------------------------------------------
# mascot
# --------------------------------------------------------------------------

class Mascot:
    """Three PNG sprites with alpha: awake / confused / sleeping."""

    STATES = ("awake", "confused", "sleeping")

    def __init__(self, sprite_dir: str = "assets/sprites") -> None:
        self.sprites: dict[str, np.ndarray | None] = {}
        self._scaled: dict[tuple[str, int], np.ndarray] = {}
        for s in self.STATES:
            p = os.path.join(sprite_dir, f"robot_{s}.png")
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                log.warning("mascot sprite missing: %s", p)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            self.sprites[s] = img

    def draw(self, img, state: str, x: int, y: int, height: int,
             bob: float = 0.0) -> None:
        sp = self.sprites.get(state)
        if sp is None:
            # Never let a missing asset show a hole: draw a simple face.
            cv2.circle(img, (x, y - height // 2), height // 2, (200, 200, 200), -1,
                       cv2.LINE_AA)
            return
        key = (state, int(height))
        scaled = self._scaled.get(key)
        if scaled is None:
            w = max(1, int(sp.shape[1] * height / sp.shape[0]))
            scaled = cv2.resize(sp, (w, int(height)), interpolation=cv2.INTER_AREA)
            self._scaled[key] = scaled
        alpha_blit(img, scaled, x, int(y + bob), anchor="cb")


class Branding:
    """Event name and institutional logos, each placed in its own corner.

    No plate behind them: supply artwork that reads on a dark background (the
    white variant of a dark logo). A missing file is skipped with a warning
    rather than leaving a hole, so the demo runs whether or not the artwork has
    arrived.
    """

    def __init__(self, entries: list[dict], text: str = "",
                 default_height: int = 56) -> None:
        self.text = text
        self.logos: list[tuple[str, np.ndarray]] = []
        for e in entries:
            path = e.get("path", "")
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                log.warning("logo missing: %s", path)
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            h = int(e.get("height", default_height))
            w = max(1, int(img.shape[1] * h / img.shape[0]))
            self.logos.append((str(e.get("pos", "top-left")),
                               cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)))

    def draw(self, img, anchors: dict[str, tuple[int, int, str]]) -> None:
        """`anchors` maps a position name to (x, y, blit anchor)."""
        for pos, logo in self.logos:
            spot = anchors.get(pos)
            if spot is None:
                continue
            x, y, anchor = spot
            alpha_blit(img, logo, x, y, anchor=anchor)

    def draw_text_at(self, img, x: int, y: int, size: int,
                     color=(245, 245, 245), anchor: str = "rm") -> None:
        if self.text:
            draw_text(img, self.text, x, y, size, color, anchor, 4)


def speech_bubble(img, text, x, y, size=44, max_w=520, colors=None):
    """Rounded speech bubble above the mascot."""
    colors = colors or {}
    tw, th = text_size(text, size, 4)
    pad = 22
    w, h = min(max_w, tw) + 2 * pad, th + 2 * pad
    x1, y1 = int(x - w // 2), int(y - h)
    rounded_rect(img, (x1, y1), (x1 + w, y1 + h), (250, 250, 250), -1, 26)
    rounded_rect(img, (x1, y1), (x1 + w, y1 + h), (40, 30, 25), 5, 26)
    tail = np.array([[x - 16, y1 + h - 2], [x + 16, y1 + h - 2], [x, y1 + h + 22]],
                    np.int32)
    cv2.fillConvexPoly(img, tail, (250, 250, 250), cv2.LINE_AA)
    draw_text(img, text, x, y1 + h // 2, size, (40, 30, 25), "cm", 0)

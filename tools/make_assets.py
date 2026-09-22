"""Generate every non-photographic asset: sounds, mascot sprites, noise decoy.

Everything is synthesised, so the repo carries no binary blobs of unknown
provenance and the demo works with no network. Re-run any time:
    .venv/bin/python tools/make_assets.py
"""
from __future__ import annotations

import math
import os
import wave

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SR = 44100


# ---------------------------------------------------------------- sounds ---

def _write_wav(path: str, data: np.ndarray) -> None:
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print("  wrote", os.path.relpath(path, ROOT), f"({data.size / SR:.2f}s)")


def _t(dur: float) -> np.ndarray:
    return np.linspace(0, dur, int(SR * dur), endpoint=False)


def _tone(freq, dur, kind="sine", decay=6.0, vol=0.5):
    t = _t(dur)
    if kind == "sine":
        w = np.sin(2 * np.pi * freq * t)
    elif kind == "tri":
        w = 2 * np.abs(2 * ((t * freq) % 1) - 1) - 1
    else:
        w = np.sign(np.sin(2 * np.pi * freq * t)) * 0.6
    return (w * np.exp(-decay * t) * vol).astype(np.float32)


def sound_pop() -> np.ndarray:
    """Soap bubble bursting: fast upward chirp + a little airy noise."""
    dur = 0.16
    t = _t(dur)
    f = 420 + 1500 * (t / dur) ** 0.4
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-26 * t)
    air = np.random.default_rng(3).normal(0, 1, t.size) * np.exp(-70 * t) * 0.25
    return (body * 0.75 + air).astype(np.float32)


def sound_detect() -> np.ndarray:
    """Friendly two-note 'boop-beep' when the robot spots a human."""
    return np.concatenate([_tone(660, 0.09, "tri", 12, 0.45),
                           _tone(990, 0.14, "tri", 9, 0.45)])


def sound_fanfare() -> np.ndarray:
    """Little victory arpeggio when the timer freezes."""
    notes = [523.25, 659.25, 783.99, 1046.50]
    parts = [_tone(f, 0.12, "tri", 7, 0.42) for f in notes]
    tail = _tone(1046.50, 0.45, "tri", 4.5, 0.42) + _tone(1568.0, 0.45, "sine", 5, 0.18)
    return np.concatenate(parts + [tail])


def sound_confused() -> np.ndarray:
    """Robot going cross-eyed: wobbly descending slide."""
    dur = 0.7
    t = _t(dur)
    f = 700 * np.exp(-1.5 * t) + 120 + 45 * np.sin(2 * np.pi * 7 * t)
    w = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return (w * np.exp(-2.6 * t) * 0.45).astype(np.float32)


# --------------------------------------------------------------- sprites ---

BODY = (250, 246, 238, 255)
EDGE = (42, 34, 30, 255)
PANEL = (58, 48, 44, 255)
AMBER = (255, 196, 60, 255)
CYAN = (90, 205, 235, 255)


def _robot_base(d: ImageDraw.ImageDraw, S: int, droop: float = 0.0):
    """Head, ears, antenna -- shared by all three states."""
    cx = S // 2
    top = int(S * 0.24)
    bot = int(S * 0.86)
    left, right = int(S * 0.16), int(S * 0.84)
    lw = max(4, S // 64)

    # antenna
    d.line([(cx, top - int(S * 0.10)), (cx, top + 6)], fill=EDGE, width=lw + 2)
    bulb = int(S * 0.045)
    by = top - int(S * 0.10)
    d.ellipse([cx - bulb, by - bulb, cx + bulb, by + bulb], fill=AMBER, outline=EDGE,
              width=lw)
    # ears
    er = int(S * 0.055)
    for ex in (left - er // 2, right + er // 2):
        d.rounded_rectangle([ex - er, (top + bot) // 2 - er * 2,
                             ex + er, (top + bot) // 2 + er * 2],
                            radius=er, fill=PANEL, outline=EDGE, width=lw)
    # head
    d.rounded_rectangle([left, top, right, bot], radius=int(S * 0.16),
                        fill=BODY, outline=EDGE, width=lw + 2)
    # face screen
    d.rounded_rectangle([left + int(S * 0.06), top + int(S * 0.08),
                         right - int(S * 0.06), bot - int(S * 0.14)],
                        radius=int(S * 0.11), fill=PANEL, outline=EDGE, width=lw)
    return cx, top, bot, left, right, lw


def sprite_awake(S: int = 512) -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, top, bot, left, right, lw = _robot_base(d, S)
    ey = top + int(S * 0.28)
    er = int(S * 0.085)
    for ex in (cx - int(S * 0.15), cx + int(S * 0.15)):
        d.ellipse([ex - er, ey - er, ex + er, ey + er], fill=CYAN, outline=EDGE,
                  width=lw)
        hr = er // 3
        d.ellipse([ex - hr + er // 3, ey - hr - er // 3,
                   ex + hr + er // 3, ey + hr - er // 3], fill=(255, 255, 255, 255))
    my = bot - int(S * 0.24)
    d.arc([cx - int(S * 0.12), my - int(S * 0.06),
           cx + int(S * 0.12), my + int(S * 0.08)], 10, 170, fill=CYAN, width=lw + 3)
    return img


def sprite_confused(S: int = 512) -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, top, bot, left, right, lw = _robot_base(d, S)
    ey = top + int(S * 0.28)
    er = int(S * 0.085)
    for ex in (cx - int(S * 0.15), cx + int(S * 0.15)):
        d.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(250, 250, 250, 255),
                  outline=EDGE, width=lw)
        # spiral eyes
        pts = []
        for i in range(70):
            a = i / 70 * math.tau * 2.6
            r = er * 0.88 * (i / 70)
            pts.append((ex + r * math.cos(a), ey + r * math.sin(a)))
        d.line(pts, fill=(230, 70, 70, 255), width=max(3, lw))
    # wavy mouth
    my = bot - int(S * 0.22)
    w = int(S * 0.13)
    pts = [(cx - w + i * (2 * w / 24), my + math.sin(i / 24 * math.tau * 2.2) * S * 0.022)
           for i in range(25)]
    d.line(pts, fill=(230, 70, 70, 255), width=lw + 3, joint="curve")
    # question marks
    for qx, qy, qs in ((right - int(S * 0.02), top + int(S * 0.02), 0.11),
                       (left + int(S * 0.01), top - int(S * 0.02), 0.08)):
        _qmark(d, qx, qy, int(S * qs), lw)
    return img


def _qmark(d, x, y, size, lw):
    d.arc([x - size // 2, y - size // 2, x + size // 2, y + size // 6],
          200, 360, fill=AMBER, width=lw + 3)
    d.line([(x + size // 6, y + size // 6), (x + size // 12, y + size // 2)],
           fill=AMBER, width=lw + 3)
    d.ellipse([x + size // 20 - lw, y + size * 3 // 5,
               x + size // 20 + lw, y + size * 3 // 5 + 2 * lw], fill=AMBER)


def sprite_sleeping(S: int = 512) -> Image.Image:
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, top, bot, left, right, lw = _robot_base(d, S)
    ey = top + int(S * 0.30)
    for ex in (cx - int(S * 0.15), cx + int(S * 0.15)):
        d.arc([ex - int(S * 0.08), ey - int(S * 0.06),
               ex + int(S * 0.08), ey + int(S * 0.07)], 200, 340,
              fill=CYAN, width=lw + 3)
    my = bot - int(S * 0.23)
    d.ellipse([cx - int(S * 0.035), my - int(S * 0.025),
               cx + int(S * 0.035), my + int(S * 0.045)], outline=CYAN, width=lw + 1)
    for i, (zx, zy, zs) in enumerate(((right - int(S * 0.04), top + int(S * 0.04), 0.10),
                                      (right + int(S * 0.04), top - int(S * 0.08), 0.07))):
        s = int(S * zs)
        d.line([(zx, zy), (zx + s, zy), (zx, zy + s), (zx + s, zy + s)],
               fill=CYAN, width=lw + 2, joint="curve")
    return img


# ----------------------------------------------------------------- decoy ---

def noise_poster(px: int = 1024, blocks: int = 10, seed: int = 11) -> Image.Image:
    """Poster 3: random colourful scribble. Looks busy, does nothing."""
    rng = np.random.default_rng(seed)
    small = rng.integers(40, 255, (blocks, blocks, 3), dtype=np.uint8)
    img = Image.fromarray(small, "RGB").resize((px, px), Image.NEAREST)
    d = ImageDraw.Draw(img)
    for _ in range(18):  # a few scribbled strokes so it reads as "drawing"
        pts = [(int(rng.integers(0, px)), int(rng.integers(0, px))) for _ in range(5)]
        d.line(pts, fill=tuple(int(v) for v in rng.integers(0, 255, 3)),
               width=int(rng.integers(8, 26)), joint="curve")
    return img


def main() -> None:
    snd = os.path.join(ROOT, "assets/sounds")
    spr = os.path.join(ROOT, "assets/sprites")
    dec = os.path.join(ROOT, "assets/decoys")
    for p in (snd, spr, dec):
        os.makedirs(p, exist_ok=True)

    print("sounds:")
    for name, fn in (("pop", sound_pop), ("detect", sound_detect),
                     ("fanfare", sound_fanfare), ("confused", sound_confused)):
        _write_wav(os.path.join(snd, f"{name}.wav"), fn())

    print("sprites:")
    for name, fn in (("awake", sprite_awake), ("confused", sprite_confused),
                     ("sleeping", sprite_sleeping)):
        p = os.path.join(spr, f"robot_{name}.png")
        fn().save(p)
        print("  wrote", os.path.relpath(p, ROOT))

    p = os.path.join(dec, "noise.png")
    noise_poster().save(p)
    print("decoy:\n  wrote", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()

"""Generate the two printable PDFs: the ArUco board and the three posters.

    .venv/bin/python tools/make_print.py

Re-run after training a new patch. Everything is A3; the three posters are
deliberately identical in size and framing, because the whole scientific point
is that the children cannot tell which one is magic by looking at it.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from reportlab.lib.pagesizes import A3, A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdfcanvas

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compositor import make_board_image  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "print")

PAPERS = {"A3": (A3, 260), "A4": (A4, 180)}   # page size -> printed square side (mm)
PAPER = "A3"
PAGE_W, PAGE_H = A3
SIDE_MM = 260


def set_paper(name: str, side_mm: float | None = None) -> None:
    global PAPER, PAGE_W, PAGE_H, SIDE_MM
    PAPER = name
    (PAGE_W, PAGE_H), default_side = PAPERS[name]
    SIDE_MM = side_mm or default_side


def _reader(bgr: np.ndarray) -> ImageReader:
    from PIL import Image
    return ImageReader(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))


def _upscale(img: np.ndarray, px: int = 2048) -> np.ndarray:
    """Print-resolution copy. Cubic keeps the pattern's structure intact."""
    return cv2.resize(img, (px, px), interpolation=cv2.INTER_CUBIC)


def _square_from(path: str) -> np.ndarray | None:
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(h, w)                       # centre crop so all posters frame alike
    y, x = (h - s) // 2, (w - s) // 2
    return _upscale(img[y:y + s, x:x + s])


def _header(c, title: str, subtitle: str = "") -> None:
    c.setFont("Helvetica-Bold", 20)
    c.drawCentredString(PAGE_W / 2, PAGE_H - 22 * mm, title)
    if subtitle:
        c.setFont("Helvetica", 11)
        c.drawCentredString(PAGE_W / 2, PAGE_H - 30 * mm, subtitle)


def _cut_marks(c, x, y, side) -> None:
    c.setLineWidth(0.4)
    c.setDash(3, 3)
    c.rect(x, y, side, side)
    c.setDash()


def board_pdf() -> str:
    path = os.path.join(OUT, f"aruco_board_{PAPER}.pdf")
    c = pdfcanvas.Canvas(path, pagesize=(PAGE_W, PAGE_H))
    _header(c, "INVISIBLE! - marker board (Mode B)",
            "Print at 100% scale. Mount on foam board. Do not crop the markers.")
    side = SIDE_MM * mm
    x = (PAGE_W - side) / 2
    y = (PAGE_H - side) / 2 - 12 * mm
    c.drawImage(_reader(make_board_image(2480)), x, y, side, side)
    _cut_marks(c, x, y, side)
    c.setFont("Helvetica", 9)
    c.drawCentredString(PAGE_W / 2, y - 8 * mm,
                        "ArUco DICT_4X4_50, ids 0=TL 1=TR 2=BR 3=BL. "
                        "The app warps the poster into the middle; keep the four "
                        "markers clean and unbent.")
    c.showPage()
    c.save()
    return path


def posters_pdf() -> str:
    path = os.path.join(OUT, f"posters_{PAPER}.pdf")
    c = pdfcanvas.Canvas(path, pagesize=(PAGE_W, PAGE_H))

    posters = [
        ("A", "assets/patches/magic_dog.png"),
        ("B", "assets/decoys/dog_photo.jpg"),
    ]
    side = SIDE_MM * mm
    x = (PAGE_W - side) / 2
    y = (PAGE_H - side) / 2 - 12 * mm

    missing = []
    for code, rel in posters:
        img = _square_from(os.path.join(ROOT, rel))
        if img is None:
            missing.append(rel)
            continue
        # No title on the page: the children must not be able to read which is which.
        c.drawImage(_reader(img), x, y, side, side)
        _cut_marks(c, x, y, side)
        c.setFont("Helvetica", 7)
        c.setFillGray(0.55)
        c.drawString(x, y - 5 * mm, f"poster {code}")   # operator key, tiny
        c.setFillGray(0)
        c.showPage()
    c.save()
    if missing:
        print("  WARNING missing, page skipped:", ", ".join(missing))
    return path


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", choices=sorted(PAPERS), default="A3")
    ap.add_argument("--side-mm", type=float, default=None,
                    help="override the printed square side")
    args = ap.parse_args()
    set_paper(args.paper, args.side_mm)
    os.makedirs(OUT, exist_ok=True)
    print(f"paper {PAPER}, square {SIDE_MM:.0f} mm, "
          f"marker ~{SIDE_MM * 0.14:.0f} mm")
    print("printables:")
    for fn in (board_pdf, posters_pdf):
        p = fn()
        print("  wrote", os.path.relpath(p, ROOT),
              f"({os.path.getsize(p)//1024} KB)")
    print("\nOperator key:  A = adversarial (magic)   B = ordinary dog")


if __name__ == "__main__":
    main()

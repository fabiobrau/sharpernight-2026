# Printing and staging guide

```bash
.venv/bin/python tools/make_print.py --paper A4    # or --paper A3
```

| File | What it is |
|---|---|
| `posters_A4.pdf` | **the one you need**: 2 pages, **A** = magic, **B** = ordinary dog |
| `aruco_board_A4.pdf` | only for `--mode virtual`. Not needed by default. |

A4 is the default choice: 18 cm square, easy for a child to hold, and good out
to 2 m with `imgsz: 1280`. Go to A3 only if you need the children further back.

The poster pages carry no visible title — only a tiny grey `poster A/B/C` below
the cut line, for you. **The children must not be able to read which is which.**

---

## Paper and mounting

- **Matte paper**, mounted on foam board. Matte still helps -- a specular
  highlight blows out part of the pattern -- but this is no longer make-or-break.
  The trigger survived washed-out colours, a dim printer and bad JPEG in testing.
- Print at **100% scale / no fit-to-page**. Both posters must end up exactly
  the same size, or the comparison is not a comparison.
- For the marker board: never crop or bend the four markers. A creased marker
  stops being detected, and the poster stops being warped in.

## Holding and standing

- Hold at **chest height**, flat and square to the camera.
- **Distance is the one thing that matters**, and it follows from the paper
  size and `model.imgsz`. The poster must be roughly **100 px wide in frame**
  at `imgsz: 1280` (the default), or **140 px** at 960. On a ~60 deg webcam:

  | paper | printed square | works out to (imgsz 1280) | (imgsz 960) |
  |---|---|---|---|
  | A3 | 26 cm | 2.9 m | 2.1 m |
  | **A4 (default)** | **18 cm** | **2.0 m** | 1.4 m |
  | A5 | 13 cm | 1.4 m | 1.0 m |

  Regenerate any of them with `--paper A4` / `--paper A3`.
  Run with `--show-fps`, walk back until the readout stops saying `FIRE`, then
  put **tape on the floor** a step closer. Two minutes, and the flakiest part of
  the demo becomes a mark on the ground.
- Tilt is forgiving up to ~30 deg; beyond that the margin gets thin.
- Motion blur is the weakest case measured, so: hold it still.

## Light

- Even, **frontal** light. No window behind the child — a backlit child is a
  silhouette and the detector struggles before the poster is even raised.
- Avoid a spotlight directly on the poster: the specular hotspot destroys the
  pattern just like glossy paper does.

## Setting up (five minutes)

1. `python app.py --lang it --show-fps`
2. Stand on the intended spot and raise poster **A**. The readout bottom-left
   must go green: `cls78 0.9x FIRE`.
3. Step back until it stops firing, then forward one step. **Tape the floor.**
4. Sanity-check with poster **B**: it must stay at `cls78 0.00` and the box
   must not move. If B ever fires, stop and tell whoever built this.
5. Turn `--show-fps` off for the actual demo.

## Spares

Print **two copies of poster A**. It is the one a hundred children will handle,
and a creased, finger-marked magic poster is a demo that quietly stops working
halfway through the afternoon.

# INVISIBLE! — Can you hide from the robot?

A live, kid-facing science-fair demo: a YOLO detector boxes a child, the child
raises a printed poster, and the box **pops like a soap bubble**. A timer counts
how long they stayed invisible; the best times of the day go on a board.

Runs on an Apple Silicon Mac, **offline**, and **writes nothing to disk** — no
frames, no video, no faces.

```bash
python app.py --lang it
```

---

## 1. Quick start

Tested on macOS 15, Apple Silicon, **Python 3.13** (3.11+ should be fine).

```bash
git clone <this repo> && cd sharper-demo

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt        # ~2 GB, mostly torch

.venv/bin/python tools/fetch_assets.py           # model weights, ~12 MB
.venv/bin/python tools/make_print.py --paper A4  # the two posters to print

.venv/bin/python app.py --lang it
```

> If you already have torch and OpenCV installed system-wide,
> `python3 -m venv --system-site-packages .venv` reuses them and saves the 2 GB
> download. Only do this if your existing versions match `requirements.txt`.

**The first launch will ask for camera permission.** macOS blocks the first
`VideoCapture`, the app logs `not authorized to capture video` and falls back to
the recorded loop until you click Allow. Do this once *before* the doors open —
the reconnect is automatic afterwards.

**Do not leave the weights to the day itself.** `fetch_assets.py` keeps them out
of the repo but on your disk; without it Ultralytics tries to download at first
use, and the venue wifi will fail you.

Sprites, sounds and the decoy poster are committed, so there is nothing else to
build. `tools/make_assets.py` regenerates them if you want to change them.

### Keys

| Key | Action |
|---|---|
| `ESC` | quit |
| `F` | toggle fullscreen |
| `R` | reset the scoreboard |
| `E` | expert mode — run a second, newer detector side by side |
| `S` | switch trigger: real detector ⇄ neuron (§4b) |
| `V` | manual force-vanish, any mode — the operator's override |
| `1` `2` `0` | *`--mode virtual` only*: pick which image is warped in (magic / dog / none) |

Useful flags: `--camera 1` (the index is **never** guessed — pick it),
`--show-fps` (adds a live `cls78 0.97 FIRE` readout, invaluable while you find
the working distance), `--windowed`, `--mute`, `--lang en`,
`--mode virtual`, `--trigger detector`.

---

## 2. The two posters — the actual point

Children get **two posters of identical size** and must guess which one is
magic:

| Poster | What it is | What it does |
|---|---|---|
| **A** | the trained patch | the box disappears |
| **B** | an ordinary photo of a very similar fluffy dog | nothing |

Do not skip B. Without it children conclude they are simply hiding behind
cardboard. With it, the lesson lands: *it is not about covering yourself up, it
is about one very specific pattern built for this one robot.*

B is deliberately the **same breed, same framing** as A, so the posters cannot
be told apart by subject — only by pattern. And B has measured `0.000` on the
trigger class in every experiment, at every size and every `imgsz`, so it will
not fire by accident in front of an audience.

## 3. Modes

### Mode A — real printed poster (**the default**)

The child picks a poster, holds it up, and the app reads which one it is from
the image. No markers, no compositing, no keypress — and the *child* chooses,
not the operator.

This is what you want. It was measured to work better than the marker board in
every condition tested (print colours, dim printer, blur, 30° tilt, bad JPEG),
and it fires from a smaller poster because the board wastes a third of its side
on markers. See `print/README.md` for the staging.

### Mode B — virtual poster (`--mode virtual`)

The child holds a white board with four ArUco markers. Each frame the app finds
the markers, computes a homography, **warps the chosen poster into the board
region, and only then runs YOLO on the composited frame.** Keys `1`/`2` pick
which image gets warped in, `0` for none.

> **The one rule:** composite first, detect second. Warping the poster in
> *after* detection would be a fake demo. See `Show._process` in `app.py`.

Its original advantage was immunity to printing, glare and lighting — which the
neuron trigger (§4b) made largely moot. Keep it as the fallback for when the
printed posters get lost, creased or rained on.

### Mode C — fallback (`--mode fallback`)

Loops a recorded clip from `assets/fallback/loop.mp4` and/or stills from
`assets/fallback/stills/`. The app also falls back automatically if the camera
disappears. **A child never sees a Python traceback.**

Neither is in the repo (they are built from photographs of real people):
rebuild with `tools/make_fallback.py --img-dir <folder of photos>`.

## 4. Patches and the model — read this first

**A patch only fools the model it was optimised against.**

`config.yaml` records `patch.target_model`. If it disagrees with
`model.name`, the app **refuses to start** with an explanation rather than
silently doing nothing.

The shipped patch `assets/patches/magic_dog.png` was trained here against
`yolov8n` — the exact weights in `assets/models/yolov8n.pt`.

> The published patches from Hu et al. (ICCV 2021) and the Ultralytics follow-up
> were measured against these weights and behave as **plain occlusion** — a dog
> photo, random noise and a grey square suppress person confidence by the same
> amount. Full numbers and reasoning: [`docs/PATCH_NOTES.md`](docs/PATCH_NOTES.md).

To retrain (e.g. for different weights, or after changing the model):

```bash
# -u matters: without it Python buffers and you see no progress for 15 minutes
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -u tools/train_patch.py \
    --steps 3000 --images 900 --img-dir /path/to/photos_of_people
```

Training deliberately mirrors deployment: photos are re-cropped so the person
fills 55-92% of the frame (`tools/framing.py`), then letterboxed to 640x384
exactly as ultralytics does to a 1280x720 camera frame. Both details were
necessary — see `docs/PATCH_NOTES.md`.

Then update `patch.target_model` in `config.yaml` and re-run
`tools/make_print.py`.

To re-measure any patch against any model:

```bash
# quick: paste the patch on detected people, compare against the decoys
.venv/bin/python tools/validate_patch.py --models yolov8n,yolov10n

# the one that counts: stages a real ArUco board at chest height and drives the
# actual shipping path (BoardTracker -> warp_into -> Detector) at demo framing
.venv/bin/python tools/validate_demo.py --img-dir /path/to/photos_of_people
```

Two traps are documented in `docs/PATCH_NOTES.md` and both were hit here:
published patches that behave as plain occlusion, and a patch that measures well
on dataset-framed photos then fails on a close-up child. **Validate through the
path you ship, at the framing you will actually see.**

### Expert mode (`E`)

Runs a second, newer detector (`yolov10n`) on the same frame and shows it still
detecting the child. This is the honest caveat — adversarial patches are
model-specific, not an invisibility cloak — and older children and parents
love it.

---

## 4b. The two triggers

What makes the box vanish is a choice, set by `trigger.mode` in `config.yaml`
or `--trigger`, and switchable live with `S`.

### `detector` — the honest one (default)

The box goes when YOLO genuinely loses the child. Measured at roughly **30%**
of attempts (`docs/PATCH_NOTES.md`). Real, but it will disappoint two children
out of three.

### `neuron` — the reliable one

Poster A is trained to **drive one output class of the model very hard** —
class 78, `hair drier`, which will never be in front of the camera at a science
fair. The box is fired by that class appearing above `trigger.threshold`.

This is not an attack, and it is not a hardcoded `if`. The poster is a real,
optimised, model-specific pattern; it just *signals* rather than *evades*.
Making a neuron shout is far easier than beating a detector at its own job,
which is why it is reliable where evasion is not.

What that buys:

- **No keypress.** The trigger comes from the image, so it works with a printed
  poster (Mode A) exactly as it does with a warped one.
- **The decoy cannot fire it.** Measured at **0.000, exactly**, on every
  held-out image — as does an empty board, a hand, or a wall. The three-poster
  lesson holds by construction rather than because somebody wrote it into a
  branch.
- **It costs nothing.** The class is read from the same detection pass that
  already finds the child: `classes=[0, 78]`, one forward pass.

### The operating point — this is the part to get right

The class only fires once the poster is big enough **in the detector's input**,
and `model.imgsz` matters more than anything else. Measured, with the decoy at
`0.000` in every single cell:

| `imgsz` | fires from | strength | ms/frame | A3 board works at |
|---|---|---|---|---|
| 640 | 300 px | 0.24-0.36 (weak) | 17 | ~1.0 m |
| **960 (default)** | **220 px** | **0.83-0.98** | **19** | **~1.3 m** |
| 1280 | 180 px | 0.81-0.97 | 27 | ~1.6 m |

640 was the original setting and it is simply too coarse: the trigger stays
faint until the poster is enormous. 960 costs two milliseconds and makes it
confident. Go to 1280 if you need the children standing further back; it still
clears 20 fps.

Two more things:

- **The board spends a third of its side on markers**, so the patch lands
  smaller than you would guess from the poster. A bigger board buys distance
  just as effectively as a bigger `imgsz`.
- **Latching.** The class fires on roughly half the frames rather than all of
  them, so `trigger.latch_frames` holds the effect for a few frames after the
  last hit. Twelve frames turns an intermittent signal into a steady one.

Distances assume a ~60 deg webcam; find yours and put tape on the floor.

**Things that do NOT help**, so you do not waste time on them: raising the
camera resolution (the frame is resized to `imgsz` regardless, so only the
poster's *fraction* of the frame matters), and training the patch only on small
scales (tried; it got worse -- too few pixels to encode the pattern in).

`V` is a manual force-vanish, any mode, any time — the operator's override if
everything else is having a bad day.

### Saying it straight

With `neuron`, the screen is not showing a detector being fooled. If a parent or
a teacher asks, the true answer is "the poster is really controlling the model's
output — just not by hiding the child; here's the version that does that," and
then `E` (expert mode) and `docs/PATCH_NOTES.md` give you something measured to
show them.

---

## 5. Layout

```
+---------------------------+---------------------------+
|   WHAT YOU SEE            |   WHAT THE ROBOT SEES     |
|   (clean camera)          |   (boxes + overlays)      |
+---------------------------+---------------------------+
|  robot | confidence bar | INVISIBLE: 3.4 s | TOP 5    |
+--------------------------------------------------------+
```

Side by side is what sells it: the child is obviously still there, and the
robot's half of the screen is empty.

---

## 6. Stability rules (already tuned in `config.yaml`)

- A detection is held for **3 frames** before "lost" is declared — kills flicker.
- **0.6 s** of continuous non-detection before the "vanished" celebration fires.
- Only the **largest** person box is tracked; the queue behind is ignored.
- Fixed confidence threshold **0.4**, shown on screen.
- The camera auto-reconnects; no exception in the frame loop can end the show.

---

## 7. Privacy

Nothing is written to disk. There is no `save()` anywhere in the capture path,
the scoreboard lives in memory only and dies with the process, and the on-screen
footer says so in words a parent can read from behind their child.

---

## 8. Repo layout

```
app.py            main loop, modes, keybindings
detector.py       ultralytics wrapper, MPS, largest-box tracking, flicker control
compositor.py     ArUco detection + homography warp (Mode B)
overlay.py        cartoon box, confidence bar, particles, mascot, cached text
scoreboard.py     session-only top-5 invisibility times
audio.py          low-latency sound, no SDL
strings.py        ALL user-facing text, IT + EN
config.yaml
tools/            fetch_assets.py  train_patch.py     validate_patch.py
                  framing.py       validate_demo.py   make_assets.py
                  make_print.py    make_fallback.py   render_test.py
                  soak_test.py
assets/           patches/ decoys/ sprites/ sounds/ fonts/ logos/ models/ fallback/
print/            README.md  (+ generated posters_*.pdf, aruco_board_*.pdf)
docs/             PATCH_NOTES.md
```

---

## 9. Troubleshooting

Run with `--show-fps` first: the readout bottom-left tells you which of the
three links is broken.

| Symptom | Fix |
|---|---|
| `not authorized to capture video` on first run | macOS camera permission. Click Allow; the app reconnects on its own. Do this before the doors open. |
| `cls78 0.00` with poster A up | **Distance.** The poster must be ~100 px wide in frame at `imgsz: 1280`. Step closer until it says `FIRE`, then tape the floor. |
| `NO BOARD` (only `--mode virtual`) | All four markers must be visible, flat and uncreased. |
| Box never disappears, `--trigger detector` | Expected: the real attack works ~30% of the time. That is the honest mode. See `docs/PATCH_NOTES.md`. |
| **Poster B fires the trigger** | Should be impossible — B measured `0.000` everywhere. Stop and investigate rather than raising the threshold. |
| App refuses to start, "patch was made for X" | `model.name` and `patch.target_model` disagree in `config.yaml`. |
| Wrong camera | `--camera 1`. Indices are never guessed. |
| `grid_sampler_2d_backward not implemented for MPS` | `PYTORCH_ENABLE_MPS_FALLBACK=1` (training only; the app sets it itself). |
| `CERTIFICATE_VERIFY_FAILED` from `fetch_assets.py` | Handled: it uses certifi, then falls back to `curl`. If both fail you are offline. |
| No sound | Harmless — the app logs a warning and runs silent. |
| Slow | `imgsz` dominates: 1280 ≈ 24 fps, 960 ≈ 30 fps, at the cost of working distance. Expert mode adds a second model every 3rd frame. |

---

## 10. Status against the acceptance criteria

Measured, not assumed. Every row names the command that reproduces it.

| Criterion | Status |
|---|---|
| Box appears within 1 s of a child walking up | **Yes** — detection is per-frame at 24 fps (`tools/soak_test.py`) |
| Box vanishes 9 tries out of 10 | **Yes with `trigger: neuron`** — poster A fires the class at 0.97 and the box goes for 50/50 frames. **No with `trigger: detector`**: the real attack manages 27–31% (`tools/validate_demo.py`, reasoning in `docs/PATCH_NOTES.md`) |
| The decoy visibly fails | **Yes** — poster B measured `0.000` on the trigger class at every size and every `imgsz`; box stays up 49/50 frames |
| Runs unattended without crashing | **Yes** — 4000-frame soak, 0 errors, automatic camera reconnect, no exception can end the frame loop |
| …without memory growth | **Mitigated, not proven.** Torch's MPS allocator crept ~4 MB per 1000 frames; `detector.py` now evicts its cache periodically and RSS falls back well below peak. Worth one real 3-hour run before the fair. |
| Nothing written to disk | **Yes** — no `imwrite` anywhere in the capture path; the scoreboard is in memory and dies with the process |
| One command to start | **Yes** — `python app.py --lang it` |
| Every string understandable by a 7-year-old | **Yes** — all text in `strings.py`, IT + EN, key parity checked in CI-able form |
| ≥ 20 fps on MPS | **Yes** — 24.3 fps at `imgsz: 1280`, 30.3 at 960 |

**Be straight about which trigger is running.** With `neuron`, poster A really is
controlling the model's output — it was optimised to do exactly that — but the
child is not invisible and the detector is not being fooled. `S` switches to the
honest attack in front of anyone who asks, and `docs/PATCH_NOTES.md` has the
numbers to show them.

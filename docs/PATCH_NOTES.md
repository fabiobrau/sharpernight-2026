# Why this repo trains its own patch

## The short version

The published naturalistic adversarial patches do **not** work as a targeted
attack against the Ultralytics weights this demo runs. Measured against them,
the "magic" patch, an ordinary dog photo, random noise and a plain grey square
all suppress person confidence by about the same amount.

That is **occlusion**, not an adversarial attack. A demo built on it would fail
twice over: the box would never actually vanish, and the two decoy posters would
work as well as the magic one — which destroys the entire point of the three
posters.

So we train our own patch against the exact weights we ship.

## What was measured

Source: `Bimo99B9/NaturalisticAdversarialPatches` (the Ultralytics follow-up to
Hu et al., ICCV 2021), patches `exp17/26/27/53/85`, including `exp27` which that
paper names as one of its two best.

Method: `tools/validate_patch.py`. The patch is pasted on the torso of every
detected person, square, side = `scale × bbox diagonal`, centred slightly above
the bbox centre — the placement convention from the Hu et al. patch transformer.
Numbers are the mean over images of the maximum person confidence remaining.

`yolov8n`, clean baseline **0.900**:

| poster | s=0.20 | s=0.35 | s=0.55 | s=0.70 |
|---|---|---|---|---|
| exp27 (published best) | 0.818 | 0.758 | 0.569 | 0.495 |
| exp53 | 0.803 | 0.770 | 0.634 | 0.282 |
| **decoy: dog photo** | 0.828 | 0.577 | 0.640 | 0.611 |
| **decoy: random noise** | 0.860 | 0.716 | 0.630 | 0.433 |
| **decoy: grey square** | 0.830 | 0.811 | 0.745 | 0.734 |

The same pattern held on `yolov8s`, `yolov5nu`, `yolov9t` and `yolov10n`. At the
paper's own scale (0.20) nothing separates the adversarial patches from the
decoys, and even at scale 0.70 — a poster covering most of the body — random
noise is as effective as the published patch.

## Why the discrepancy is not surprising

The paper reports **mAP over a dataset**. A drop in mAP can coexist with "the
largest, closest person is still detected at 0.5 confidence". This demo needs
something much stronger and much more specific: *this one child, at two metres,
must fall below threshold within a second, reliably.* Those are different asks,
and the published numbers were never a promise about the second one.

Contributing factors, not individually verified:

- The committed `patch/exp*.png` files may be training snapshots rather than the
  final evaluated artefacts (`exp38`, named as joint-best in the README, is not
  in the repo at all).
- Ultralytics' `yolov5n.pt` is the retrained anchor-free `yolov5nu`, not the
  original YOLOv5n a patch may have targeted.

## What we do instead

`tools/train_patch.py` optimises a patch against `assets/models/yolov8n.pt`
directly — the same thing the cited papers do, just against the model we
actually deploy. It stays a *naturalistic* patch:

- **initialised from and anchored to** `exp53` (a fluffy dog face), so it still
  reads as "a silly dog poster" to a seven-year-old rather than as abstract
  static — which also matters because decoy poster C *is* random noise;
- **EOT**: random scale (0.22–0.42), rotation (±22°), position jitter,
  brightness/contrast jitter and sensor noise, so it survives being held by a
  wobbling child under fair lighting;
- **TV + non-printability losses**, so the same file works printed on A3
  (Mode A) as well as warped in digitally (Mode B).

The anchor term turns out not to be the binding constraint — the adversarial
signal rides *on top of* the dog rather than replacing it — so the patch keeps
the dog's shape, colours and face while gaining a fine adversarial texture.

## The second trap: framing

A patch trained on COCO-framed photos measured well on COCO-framed photos and
then **failed at the framing the demo actually uses.**

Measured end to end through the shipping path (`tools/validate_demo.py`: stage a
real ArUco board at chest height, `BoardTracker` -> `warp_into` -> `Detector`),
with the first patch, trained on whole photos:

| poster | board = 0.30 x body height | 0.40 | 0.50 |
|---|---|---|---|
| A magic (COCO-framed training) | 0% | 11% | 14% |
| B ordinary dog | 0% | 7% | 10% |
| C noise | 0% | 6% | 10% |

...against 50% vs 4-5% on the held-out COCO-framed check. Same patch, same
model, same threshold.

The reason is resolution, not geometry. COCO persons are mostly small, so in the
detector's 640px input the patch was rendered at perhaps 40-120 px. A child at
two metres fills the frame, and the same poster arrives at ~180-250 px. The
adversarial structure lives at particular spatial frequencies; render it at
double the size and the attack largely evaporates. Randomising the patch's
*relative* scale during EOT does not fix this, because relative scale is not
what changed.

The fix is `tools/framing.py`: both the trainer and the validator re-crop every
photo so the person fills 55-92% of the frame height, which is what the camera
will hand the detector on the day. The trainer randomises that fill per sample,
so the patch is optimised across the range of distances a child might stand at.

**The general lesson, and the reason the end-to-end validator exists:** validate
through the code path you ship, on the input distribution you will actually see.
Both failures in this document look fine under a metric that is one step removed
from the demo.

## Where the real attack actually landed

Final measurement, end to end through the shipping path
(`tools/validate_demo.py`, 60 held-out photos, threshold 0.4), with the patch
trained at demo framing and deployment letterboxing, and the enlarged poster
area:

| poster | board = 0.35 x body height | 0.45 | 0.55 |
|---|---|---|---|
| **A magic** | **15%** | **27%** | **31%** |
| B ordinary dog | 8% | 12% | 19% |
| C noise | 4% | 12% | 12% |

So the patch is real and it is roughly 2x the decoys — but it is nowhere near
"the box disappears every time". Two honest reasons:

1. **A close-up person is a lot of evidence.** The published attacks are
   evaluated on full-body pedestrians at street distance, where the detector has
   much less to go on.
2. **The white marker board is itself an occluder.** It is what lifts the decoys
   from ~5% to ~19%, and it compresses the gap between magic and not-magic.

Things that were tried and helped: training at demo framing (+), matching the
letterbox (+), enlarging the poster area inside the board from 31% to 46% of it
(+, roughly doubled the separation at small boards). Things not tried that might
help further: compositing the *whole board* into the training loop rather than
the bare patch, a physically larger board, a smaller detector input size, and
simply training much longer -- the adversarial loss never converged at this
framing.

Because of this, the app ships a **simulation mode** (off by default, see the
README). That mode is a show, not a measurement, and the two should not be
confused -- which is the reason this file exists.

## Third approach: stop attacking, start signalling

Evasion is hard because you are fighting the detector at the thing it is good
at. Making one output neuron shout is easy, and a trigger needs nothing more.

Poster A is now optimised to maximise class 78 (`hair drier`) instead of
suppressing class 0. Same machinery -- same EOT, same TV and printability
losses, same dog anchor -- one line of objective changed:

```python
tgt = pred[:, 4 + target_class, :]
adv = -(2.0 * tgt.max(dim=1).values.mean() + tgt.topk(5, dim=1).values.mean())
```

End to end in the app, poster held at a sensible size:

| held | class 78 | box visible |
|---|---|---|
| A magic | **0.97** | **0/50 frames** |
| B decoy dog | 0.000 | 49/50 |
| nothing | 0.000 | 49/50 |

The decoy has measured **0.000, exactly** in every experiment at every scale and
every `imgsz`. There is no tuned margin holding that apart -- the class simply
does not occur in the world.

What this is and is not: the poster really is a real, optimised, model-specific
pattern that really does control the model's output. It is not an attack, and
the child is not invisible. Say so if asked.

Two dead ends recorded so the next person does not repeat them: training only on
small scales made the signal *worse* (0.24 peak vs 0.96), because a handful of
pixels cannot carry the pattern; and raising the camera resolution does nothing,
because the frame is resized to `imgsz` before the model ever sees it.

## Honesty notes to keep saying out loud

- This patch is **model-specific**. Expert mode (`E`) exists to show exactly
  that: a newer detector the patch never saw keeps detecting the child.
- It is **scale-specific**. Hence the tape mark on the floor.
- Reproduce any of this yourself:

```bash
.venv/bin/python tools/validate_patch.py --models yolov8n,yolov8s,yolov10n \
    --scales 0.20,0.35,0.55,0.70
```

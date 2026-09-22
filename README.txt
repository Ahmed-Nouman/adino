# Adino — Cube Stacking Monitor

A real-time vision system that watches a worker stack coloured cubes, checks the
stacking order, and tells them what to do next.

A YOLO model detects the cubes; a rule-based layer on top groups them into
stacks, reads each stack bottom → top, validates it against the required
sequence, and produces guidance such as *"place the green cube on top of the
red cube"* or *"remove the white cube at position 1 — green belongs there"*.

Required order (bottom → top): **red → green → black → white**

---

## How it works

```
camera frame
   │
   ▼
YOLO detection ──► colour check ──► group into stacks ──► hidden-cube inference
                                                                  │
                    guidance ◄── validate order ◄── regroup ◄─────┘
```

1. **Detection** — a YOLO model (`best.pt`) finds cubes and their colour class.
2. **Colour check** — a per-pixel HSV test corrects a common YOLO error: under a
   warm colour cast, the white cube can be classified as red. A box is relabelled
   white/black only if YOLO's own colour is essentially absent from its pixels.
   "Black" boxes whose pixels are bright (e.g. a dark-blue board) are rejected.
3. **Grouping** — cubes form a stack when their centres line up horizontally
   **and** their boxes are a similar size. Apparent size falls with distance, so a
   cube standing further back on the table is not mistaken for part of a stack.
   Single cubes are reported as *loose*, not as one-cube stacks.
4. **Hidden-cube inference** — YOLO often misses the bottom cube of a stack
   (its top face is hidden, so it looks unlike a loose cube). The system checks
   the region one cube-step below each stack; if a cube colour fills every cell
   of a 3×3 grid there, a cube is inferred. Inferred cubes are drawn with a
   **dashed box** and a `?` after the label.
5. **Validation & guidance** — the stack whose bottom cube is red is the build
   site; its order is checked and the next action is shown in a banner.

## Two live modes

| Script | YOLO runs | Use when |
|---|---|---|
| `detect_stack_live.py` | every frame | debugging, tuning, maximum responsiveness |
| `detect_stack_live_after_action.py` | only after the scene settles | normal operation |

The **motion-gated** version runs a cheap motion check on every frame and only
runs YOLO once a placement is finished:

| State | Meaning | YOLO |
|---|---|---|
| `IDLE` | scene still, last result shown | only a heartbeat every 5 s |
| `MOVING` | hand/cube moving, banner shows *PLACING* | no |
| `SETTLING` | motion stopped, waiting 0.7 s to be sure | no |
| `CONFIRMING` | checking until 3 results agree | yes |

Guidance is therefore never given on a half-finished placement, a cube held in
the hand is never evaluated, and the log gets one entry per placement. In
testing, YOLO ran on roughly 2 % of frames.

Both versions share identical detection and stacking logic, so they give the
same result whenever YOLO runs.

## Repository layout

```
adino/
├── model/
│   └── best.pt                    # YOLO weights (see "Model weights" below)
├── scripts/                       # adjust to your actual folder name
│   ├── detect_stack_live.py        # live monitor, YOLO every frame
│   ├── detect_stack_live_after_action.py  # live monitor, motion-gated
│   └── detect_stack_order_image.py       # early single-image version (see note)
├── result/                        # generated output (git-ignored)
├── requirements.txt
└── README.md
```

The live scripts resolve paths relative to their own folder:
`MODEL_PATH = '../model/best.pt'` and `OUTPUT_FOLDER = '../result'`.

## Setup

**Hardware**
- Luxonis **OAK-D Pro** (default), or any camera OpenCV can open (USB webcam,
  RealSense RGB stream via V4L2)
- A GPU is optional but recommended for the every-frame version

**Install**

```bash
git clone git@github.com:Ahmed-Nouman/adino.git
cd adino
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies: `ultralytics`, `opencv-python`, `numpy`, and `depthai` (v3, only
needed for the OAK camera).

**Model weights** — place the trained model at `model/best.pt`. If the repo uses
Git LFS for `*.pt` files, run `git lfs pull` after cloning.

## Usage

```bash
cd scripts

# Motion-gated live monitor (recommended), OAK-D Pro
python ko_yolo_stack_live_gated.py

# Every-frame live monitor
python ko_yolo_stack_live.py

# Using a webcam instead of the OAK
python ko_yolo_stack_live_gated.py --source webcam --cam 0

# Analyse a single image (full pipeline, writes an annotated image + report)
python ko_yolo_stack_live.py --image path/to/image.jpg
```

> **Note:** `ko_yolo_stack_order.py` is the early standalone single-image script.
> It predates the colour check and hidden-cube inference. For analysing images
> with the current logic, use `ko_yolo_stack_live.py --image` instead.

### Keyboard controls

| Key | Action | Modes |
|---|---|---|
| `q` / `Esc` | quit | both |
| `p` | pause / resume | both |
| `s` | save snapshot (annotated image + text report) | both |
| `d` | print detections with HSV values, near-miss detections, and probe decisions | both |
| `a` | draw the work area for motion detection (`Enter` confirm, `c` cancel) | gated |
| `r` | force a re-check now | gated |

### On-screen markers

| Marker | Meaning |
|---|---|
| Box in the cube's own colour | YOLO detection |
| Dashed box, label ends in `?` | cube inferred from colour, not detected by YOLO |
| Label ends in `*` | colour check overrode YOLO's class |
| Magenta frame + `[WRONG]` / `[EXTRA]` | cube in the wrong position |
| Yellow frame + `[CHECK]` | irregular gap or spread in the stack |
| `(base)` / `(loose)` | single cube: the red base, or a cube waiting on the table |

Banner colours: green = stack complete, orange = in progress, red = error,
grey = placing / not ready.

## Output

Written to `result/` (git-ignored):

- `stack_log_<session>.txt` — one line per accepted state change, with guidance
- `snapshot_<time>.jpg` + `snapshot_<time>_stack.txt` — saved with the `s` key
- For `--image` runs: the annotated image and a `<name>_stack.txt` report

## Configuration

All settings are constants at the top of each script.

**Detection & stacking**

| Parameter | Default | Purpose |
|---|---|---|
| `CONF` | `0.25` | YOLO confidence threshold |
| `EXPECTED_ORDER` | red, green, black, white | required order, bottom → top |
| `SAME_STACK_X_FRAC` | `0.6` | max horizontal centre offset, as a fraction of cube width |
| `SIZE_RATIO_TOL` | `0.25` | box widths must agree within 25 % to share a stack |
| `SHOW_ANGLE` | `False` | per-cube orientation (unreliable on flat faces) |

**Colour check & hidden-cube inference**

| Parameter | Default | Purpose |
|---|---|---|
| `COLOR_CHECK` | `True` | enable the YOLO colour correction |
| `RELABEL_OWN_MAX` | `0.2` | relabel only if YOLO's colour covers < 20 % of pixels |
| `BLACK_V_MAX` | `80` | reject "black" boxes brighter than this |
| `INFER_HIDDEN` | `True` | enable inference of cubes hidden under a stack |
| `PROBE_CELL_MIN_FRAC` | `0.5` | every grid cell must match the colour at least this much |
| `MANUAL_WB_K` | `4000` | lock OAK white balance (K); lower if image looks orange |

**Motion gating** (gated version only)

| Parameter | Default | Purpose |
|---|---|---|
| `MOTION_AREA_FRAC` | `0.01` | fraction of the work area that must change to count as motion |
| `MOTION_PIXEL_THRESH` | `25` | per-pixel colour change that counts as changed |
| `SETTLE_SECONDS` | `0.7` | stillness required before checking |
| `CONFIRM_FRAMES` | `3` | YOLO results that must agree |
| `HEARTBEAT_SECONDS` | `5.0` | re-check a still scene this often |
| `WORK_AREA` | `None` | motion region as frame fractions; set live with `a` |

## Troubleshooting

**No cubes detected.** Press `d`. If near-miss detections appear below `CONF`,
lower `CONF`. If nothing appears at all, the scene is too far from the training
data — check lighting and camera angle, and avoid a bright window behind the
work area.

**White cube shown as red, or colours look orange.** Auto white balance is
compensating for daylight in the frame. Lock it with `MANUAL_WB_K`, and keep the
camera pointed away from windows.

**Gated version never leaves `MOVING`, or re-checks constantly.** Camera noise
or auto-exposure is registering as motion. Press `d` on a still scene to see its
motion level, then set `MOTION_AREA_FRAC` comfortably above it. Setting a work
area with `a` also helps.

**A cube is inferred where there isn't one.** Press `d` to see each probe's
grid values, then raise `PROBE_CELL_MIN_FRAC` or `PROBE_MIN_FRAC`.

## Known limitations

- Stacking is judged from colour and 2D geometry. Unusual arrangements — e.g. a
  cube placed directly behind and exactly aligned with a stack — can still be
  misread. The OAK-D Pro's depth stream could resolve these and is the natural
  next step.
- Any loose red cube is treated as the stack base. With a spare red cube on the
  table, restrict the scene or add a work-area filter for detections.
- In the gated version, a hand held perfectly still over the stack for longer
  than `SETTLE_SECONDS` is evaluated with the hand in view. The scene is
  re-checked when the hand leaves, so the final result is correct, but the log
  gets one extra entry.
- The most robust fix for missed bottom cubes is retraining with images of full
  stacks from the deployed camera position; inference is a practical workaround.

## Author

Ahmed Nouman

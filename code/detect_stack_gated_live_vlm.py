#!/usr/bin/env python3
"""
LIVE cube stacking assistant - YOLO detects, a VLM (Ollama) checks and instructs.

After EVERY settle (the scene was still for SETTLE_SECONDS after motion, and
CONFIRM_FRAMES YOLO results agreed), a folder result/settles/settle_<time>/ gets:
     1_raw.jpg               the clean camera frame
     2_yolo_only.jpg         plain YOLO boxes (no colour check, no inference)
     3_yolo_geometric.jpg    YOLO + colour check + stack grouping + hidden cubes
     4_final_vlm.jpg         3 + the VLM result: reading, verdict, instructions
     result.json             all readings bottom -> top, comparison, instructions
  1. the frame is saved and YOLO-only / YOLO+geometry images are drawn
  2. YOLO + geometry is compared with the goal. Only if that shows an ANOMALY
     (wrong order, a goal cube not seen, a gap in a stack) is the VLM called;
     otherwise the instructions come from code at once (VLM_ON_ANOMALY_ONLY).
     The VLM gets the frame WITH the YOLO boxes drawn on it and reports what is
     on the table, bottom -> top (LOOK). It complements YOLO: cubes YOLO missed
     are added, wrongly guessed cubes are dropped. Its reading is only refused
     if it contradicts a confident YOLO box or lists a cube twice - then it is
     told why and asked once more, and YOLO's reading is used if it still fails.
  3. code checks the order of the final reading against EXPECTED_ORDER
     (complete / in_progress / wrong + the exact anomaly), for YOLO's reading,
     the VLM's reading and the final one, so they can be compared.
  4. the same model (or an optional --planner-model) gives every move. Code
     only replays the moves to check they are possible and finish the goal; if
     not, the reason goes back and it retries. Valid moves become instructions.
  5. 4_final_vlm.jpg and result.json are written.
  The GUI shows the verdict, the instructions and what the VLM added/disagreed on.

Model (Ollama): Qwen3-VL does both steps - no separate planner needed.
    ollama pull qwen3-vl:8b        (qwen3-vl:4b / :2b faster, :32b more accurate)

Motion gating (unchanged):
    IDLE        scene still, last result shown, no YOLO (except a heartbeat)
    MOVING      hand / cube moving - instructions stay up, boxes are hidden
    SETTLING    motion stopped, waiting SETTLE_SECONDS to be sure
    CONFIRMING  YOLO runs until CONFIRM_FRAMES results agree, then -> IDLE
A heartbeat re-check only counts as a new settle if YOLO's reading changed.

Labels (same on screen and in the image the VLM sees):
    S1-2 green  -> stack 1 (stacks numbered left to right), 2nd cube from the bottom
    green       -> a single cube standing on the table
    dashed box  -> cube guessed from colour under a stack, not detected by YOLO

Goal (target order) - not hard-coded:
    default            at start the window asks you to build the TARGET stack
                       and press SPACE; check the reading, ENTER to accept
    --goal-image F     read the goal from a photo of the target stack
    --goal a,b,c,d     give it directly, bottom -> top
    Saved in result/goal/goal_<time>/ (photo, annotated photo, goal.json).

Sources:
    python detect_stack_live_vlm.py                        # OAK-D Pro (default)
    python detect_stack_live_vlm.py --source webcam --cam 0
    python detect_stack_live_vlm.py --image img.jpg --goal-image goal.jpg   # single image
    python detect_stack_live_vlm.py --no-vlm                  # detections only
    python detect_stack_live_vlm.py --vlm-model qwen3-vl:4b

Keys (live mode):
    q / ESC : quit
    s       : save snapshot (view + detection report + VLM answer)
    p       : pause / resume
    i       : toggle the info line (FPS, gate state, motion, YOLO / VLM load)
    d       : print every detection with its HSV colour values (for tuning)
    a       : draw the work area for motion detection (ENTER to confirm,
              c to cancel; an empty box resets to the full frame)
    r       : force a YOLO re-check now
    v       : treat the current scene as a new settle and ask the VLM again
"""

import argparse
import base64
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
import zlib
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = '../model/best.pt'
OUTPUT_FOLDER = '../result'

CONF = 0.25
DEVICE = None                       # None = auto, or 'cuda:0' / 'cpu'
RESIZE = None                 # frame size fed to YOLO; None = native
CAMERA_FPS = 30
SHOW_ANGLE = False                  # True -> also compute per-cube orientation

# Lock the OAK's white balance (Kelvin). With a bright window in view, auto
# white balance "corrects" for the blue daylight and turns the whole room
# orange - which is what makes the white cube look red-orange to the model.
# Image still too orange -> lower this. Too blue -> raise it. None = auto.
MANUAL_WB_K = 4000

# Colour check: a cheap HSV sanity test on each box after YOLO. Saturation
# separates white/black from red/green reliably even under a colour cast,
# where hue does not. If YOLO calls an unsaturated cube 'red' or 'green', it
# is relabelled white (bright) or black (dark). Tune with the 'd' key.
COLOR_CHECK = True
ACHROMATIC_S_MAX = 130       # median saturation (0-255) below this -> white/black
WHITE_V_MIN = 100            # achromatic and at least this bright -> white
BLACK_V_MAX = 80             # a 'black' box brighter than this is rejected
RELABEL_OWN_MAX = 0.2        # relabel only if YOLO's colour covers < 20 % of pixels
ROI_INNER_FRAC = 0.5         # sample the central 50 % of the box only
ACHROMATIC = ('white', 'black')

# Hidden-cube inference. YOLO often misses the bottom cube of a stack: with a
# cube on top its top face is hidden and only a flat front face shows, which
# looks unlike the loose cubes it was trained on. So below the bottom detected
# cube of every group, one cube-step down, the pixels are classified by colour.
# If they clearly match a cube colour, a cube is inferred there. Any colour is
# accepted, not only the expected one, so a wrong base is still reported.
INFER_HIDDEN = True
PROBE_MIN_FRAC = 0.6         # fraction of probe pixels that must match a colour
# A cube hidden under a stack sits directly beneath it, so its colour fills the
# whole probe region edge to edge. A cube standing IN FRONT of the stack only
# covers part of it. The probe is split into a PROBE_GRID x PROBE_GRID grid and
# every cell must match the colour, not just the average.
PROBE_GRID = 3
PROBE_CELL_MIN_FRAC = 0.5
# Never infer a cube where YOLO already detected something: those pixels
# belong to that object, and the size check in grouping decides whether it is
# really part of the stack.
PROBE_MAX_OVERLAP = 0.3      # fraction of the probe box covered by a detection
# Pixel colour masks (OpenCV HSV: H 0-180, S/V 0-255). Measured on the live
# camera: red base H~177 S~255; wood H~6-9 S~90-165; green H~44 S~133;
# black V~38; white S~48 V~241. The red hue band excludes wood's H~9.
PROBE_RED_H = (6, 170)       # red if H <= 6 or H >= 170 ...
PROBE_RED_S_MIN = 150        # ... and saturation at least this
PROBE_GREEN_H = (35, 85)
PROBE_GREEN_S_MIN = 90
PROBE_BLACK_V_MAX = 50
PROBE_WHITE_S_MAX = 60
PROBE_WHITE_V_MIN = 150

DEBUG_CONF = 0.05            # 'd' key also lists near-miss detections down to this

# Goal stack, bottom -> top. Not hard-coded: set at start-up from a captured
# photo of the target stack (default), --goal-image FILE, or --goal a,b,c.
EXPECTED_ORDER = []
GOAL_SOURCE = None
MODEL_COLOURS = []           # colour names of the YOLO classes (set in main)

MIN_STACK_SIZE = 2

SAME_STACK_X_FRAC = 0.6      # centre offset, as a fraction of mean cube width
SIZE_RATIO_TOL = 0.25        # box widths must agree within 25 %

# Side faces. A cube turned slightly shows two faces, and YOLO may box each
# one. Boxes narrower than SIDE_FACE_ASPECT (width / height) are side faces.
# Same-colour boxes side by side (overlapping vertically by MERGE_Y_OVERLAP of
# the shorter one, horizontal gap below MERGE_X_GAP of the wider one) are the
# same cube: the most confident box is kept. Any two boxes covering the same
# area (MERGE_CONTAIN of the smaller one) are also one cube.
SIDE_FACE_ASPECT = 0.6
MERGE_Y_OVERLAP = 0.6
MERGE_X_GAP = 0.2
MERGE_CONTAIN = 0.7
SIDE_FACE_X_GAP = 0.3        # a side face joins a stack within this gap (x width)

GAP_MULTIPLE_WARN = 1.6      # gap > 1.6 * the typical gap -> cube probably missing
X_SPREAD_WARN_FRAC = 0.8     # members spread wider than this -> stacks may be merged

LABEL_NOISE_TOKENS = ('cube', 'cubes', 'block', 'blocks', 'box', 'boxes')


# ---- Motion gating ----
# Motion = fraction of pixels in the work area whose colour differs by more
# than MOTION_PIXEL_THRESH from an anchor frame (downscaled, blurred).
MOTION_SCALE = 0.25          # downscale before differencing - fast and denoised
MOTION_PIXEL_THRESH = 25     # colour change (0-255, any channel) that counts
MOTION_AREA_FRAC = 0.01      # > 1 % of the work area changed -> motion
MOTION_MIN_FRAMES = 2        # IDLE: consecutive changed frames -> MOVING
SETTLE_SECONDS = 0.7         # stillness required before YOLO runs
CONFIRM_FRAMES = 3           # YOLO results that must agree before accepting
CONFIRM_MAX_FRAMES = 15      # give up waiting for agreement; take majority
HEARTBEAT_SECONDS = 5.0      # re-check a still scene this often anyway
# Work area for motion detection as frame fractions (x1, y1, x2, y2), or None
# for the full frame. Set it live with the 'a' key. Restricting it stops people
# walking past in the background from pausing the monitor.
WORK_AREA = None

# ---- VLM check (Ollama) ----
VLM_ENABLED = True               # --no-vlm turns it off
VLM_MODEL = 'qwen3-vl:8b'        # must be a VISION model (qwen3-vl, gemma4, ...)
VLM_PLANNER_MODEL = None         # None = VLM_MODEL also plans the moves
VLM_BACKEND = 'api'              # 'api' = Ollama REST on OLLAMA_URL, 'cli' = `ollama run`
OLLAMA_URL = 'http://localhost:11434'
VLM_TIMEOUT = 180                # seconds; the first call also loads the model
VLM_KEEP_ALIVE = '15m'           # keep the models in GPU memory between checks
VLM_MAX_SIDE = 768               # longest image side sent to the VLM (pixels)
VLM_CROP_TO_WORK_AREA = True     # send only the work area if one is set ('a' key)
VLM_ANNOTATE = False             # False: VLM sees the RAW image and no YOLO reading
                                 # (independent check). True: boxes + reading shown
# Colours that pixel colour cannot separate from the background (under a warm
# colour cast the white cube and the white wall have the same hue/saturation):
# VLM boxes for these are checked on size and position only, no colour fit.
NO_COLOUR_CHECK = ('white',)
# Call the VLM only when YOLO + geometry shows an anomaly against the goal
# (wrong order, a goal cube not seen anywhere, a gap in a stack). Normal
# progress and a complete stack are handled by code alone. False = every settle.
VLM_ON_ANOMALY_ONLY = True
VLM_GROUND = True                # ask the VLM for boxes of cubes YOLO missed
VLM_GROUND_SIDE = 1000           # square image size for that request
VLM_LOOK_RETRIES = 1            # re-ask LOOK when its reading contradicts sure facts
VLM_PLAN_RETRIES = 2             # re-ask when a plan is impossible or misses the goal
# A YOLO box (not a guessed one) at least this confident must also be in the
# VLM's reading; otherwise the VLM is told to look again.
VLM_TRUST_CONF = 0.6
# Every goal colour is ONE physical cube - a reading listing a colour twice is wrong.
UNIQUE_COLOURS = True
VLM_OPTIONS = {
    'temperature': 0.1,          # near-deterministic answers
    'num_ctx': 4096,
    'num_predict': 1500,         # room for the JSON even if the model thinks first
}

# Box colour for each cube class (BGR). Keys are normalised labels, so
# 'red_cube', 'Red Block' etc. all map to 'red'. Add entries for new classes.
CUBE_COLORS = {
    'red':    (0, 0, 255),
    'green':  (0, 200, 0),
    'blue':   (255, 80, 0),
    'yellow': (0, 230, 255),
    'orange': (0, 140, 255),
    'purple': (200, 0, 150),
    'black':  (0, 0, 0),
    'white':  (255, 255, 255),
}
# Fallback colours for classes not listed above (picked deterministically).
FALLBACK_COLORS = [(200, 130, 0), (130, 0, 200), (0, 180, 180), (180, 180, 0)]

# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def get_orientation_angle(roi):
    """Rotation angle of a cube inside a cropped region."""
    if roi is None or roi.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0

    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    angle = rect[-1]
    width, height = rect[1]
    if width < height:
        angle = 90 + angle
    return round(angle, 2)


def roi_hsv(image, x1, y1, x2, y2):
    """Median H, S, V of the central part of a box (skips edges/background)."""
    w, h = x2 - x1, y2 - y1
    mx = int(w * (1 - ROI_INNER_FRAC) / 2)
    my = int(h * (1 - ROI_INNER_FRAC) / 2)
    roi = image[max(0, y1 + my):max(0, y2 - my), max(0, x1 + mx):max(0, x2 - mx)]
    if roi.size == 0:
        return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    return tuple(float(v) for v in np.median(hsv, axis=0))


def verify_color(cube, names_by_color, image):
    """
    Colour sanity checks on a YOLO box, using per-pixel masks on the centre:
      - relabel to white/black only if YOLO's own colour is essentially absent
        (< RELABEL_OWN_MAX of pixels) AND most pixels are unsaturated. A
        median-saturation threshold alone flipped green (S ~130) to white;
      - a 'black' box whose pixels are bright is rejected (e.g. a dark-blue
        board at the frame edge).
    YOLO's own white/black decisions are otherwise left alone.
    """
    if cube['hsv'] is None:
        return
    _, _, v = cube['hsv']
    name = normalize_label(cube['label'])

    if name == 'black' and v > BLACK_V_MAX:
        cube['reject'] = True
        return
    if name in ACHROMATIC:
        return

    x1, y1, x2, y2 = cube['x1'], cube['y1'], cube['x2'], cube['y2']
    mx = int((x2 - x1) * (1 - ROI_INNER_FRAC) / 2)
    my = int((y2 - y1) * (1 - ROI_INNER_FRAC) / 2)
    roi = image[max(0, y1 + my):max(0, y2 - my), max(0, x1 + mx):max(0, x2 - mx)]
    if roi.size == 0:
        return
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(int)
    masks = color_masks(hsv)
    if name not in masks:                     # a colour we have no mask for
        return

    own = float(masks[name].mean())
    unsaturated = float((hsv[..., 1] < ACHROMATIC_S_MAX).mean())
    if own >= RELABEL_OWN_MAX or unsaturated < 0.5:
        return

    target = 'white' if v >= WHITE_V_MIN else 'black'
    new_label = names_by_color.get(target)
    if new_label is not None:
        cube['label'] = new_label
        cube['recoloured'] = True


def class_names_by_color(model):
    """{'red': 'red_cube', ...} built from the model's own class names."""
    return {normalize_label(n): n for n in model.names.values()}


def detect_cubes(model, image, conf=None, checks=True):
    """Run YOLO and return a flat list of cube dicts."""
    h_img, w_img = image.shape[:2]
    names_by_color = class_names_by_color(model)
    cubes = []

    results = model(image, conf=CONF if conf is None else conf,
                    device=DEVICE, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])

            cube = {
                'label': model.names[cls_id],
                'yolo_label': model.names[cls_id],
                'recoloured': False,
                'inferred': False,
                'reject': False,
                'conf': float(box.conf[0]),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) / 2.0,
                'cy': (y1 + y2) / 2.0,
                'w': float(x2 - x1),
                'h': float(y2 - y1),
                'angle': 0.0,
                'hsv': roi_hsv(image, x1, y1, x2, y2),
            }

            if COLOR_CHECK and checks:
                verify_color(cube, names_by_color, image)
                if cube['reject']:
                    continue

            if SHOW_ANGLE:
                pad = 5
                roi = image[max(0, y1 - pad):min(h_img, y2 + pad),
                            max(0, x1 - pad):min(w_img, x2 + pad)]
                cube['angle'] = get_orientation_angle(roi)

            cubes.append(cube)

    return cubes


PROBE_LOG = []               # last frame's probe decisions, printed by the 'd' key


def color_masks(hsv):
    """Per-pixel boolean masks for each cube colour. hsv: (..., 3) int array."""
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return {
        'red': ((H <= PROBE_RED_H[0]) | (H >= PROBE_RED_H[1]))
               & (S >= PROBE_RED_S_MIN) & (V >= 60),
        'green': (H >= PROBE_GREEN_H[0]) & (H <= PROBE_GREEN_H[1])
                 & (S >= PROBE_GREEN_S_MIN) & (V >= 50),
        'black': V < PROBE_BLACK_V_MAX,
        'white': (S < PROBE_WHITE_S_MAX) & (V >= PROBE_WHITE_V_MIN),
    }


def classify_patch(image, x1, y1, x2, y2):
    """
    Colour of a region, or None, plus the per-cell fractions for debugging.

    The region is trimmed 10 % at each side and 15 % at the bottom (the top is
    kept: the hidden cube must start right under the cube above), split into a
    grid, and a colour only wins if it fills EVERY cell. Uses per-pixel masks,
    not a median hue, because red's hue wraps around 0/180.
    """
    w, h = x2 - x1, y2 - y1
    rx1, rx2 = x1 + int(0.10 * w), x2 - int(0.10 * w)
    ry1, ry2 = y1 + int(0.05 * h), y2 - int(0.15 * h)
    roi = image[max(0, ry1):max(0, ry2), max(0, rx1):max(0, rx2)]
    if roi.shape[0] < PROBE_GRID or roi.shape[1] < PROBE_GRID:
        return None, {}

    masks = color_masks(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(int))
    rh, rw = roi.shape[:2]
    n = PROBE_GRID

    best, best_frac, cells_by_color = None, 0.0, {}
    for name, m in masks.items():
        cells = [float(m[r * rh // n:(r + 1) * rh // n,
                         c * rw // n:(c + 1) * rw // n].mean())
                 for r in range(n) for c in range(n)]
        cells_by_color[name] = cells
        overall = float(m.mean())
        if overall >= PROBE_MIN_FRAC and min(cells) >= PROBE_CELL_MIN_FRAC \
                and overall > best_frac:
            best, best_frac = name, overall
    return best, cells_by_color


def _overlap_frac(box, cube):
    """Fraction of `box` (x1, y1, x2, y2) covered by the cube's box."""
    x1, y1, x2, y2 = box
    ix = max(0, min(x2, cube['x2']) - max(x1, cube['x1']))
    iy = max(0, min(y2, cube['y2']) - max(y1, cube['y1']))
    area = max(1, (x2 - x1) * (y2 - y1))
    return ix * iy / area


def infer_hidden_cubes(image, groups, names_by_color, detected=()):
    """
    Look one cube-step below the bottom cube of every group that doesn't
    already start with the base colour. Returns a list of inferred cubes;
    keeps stepping down until the pixels stop looking like a cube or the
    base colour is found.
    """
    PROBE_LOG.clear()
    if not INFER_HIDDEN:
        return []

    # no goal yet (goal capture): probe under every stack, any colour
    base = normalize_label(EXPECTED_ORDER[0]) if EXPECTED_ORDER else None
    max_cubes = len(EXPECTED_ORDER) or len(MODEL_COLOURS) or 4
    h_img = image.shape[0]
    inferred = []

    for group in groups:
        bottom = group['members'][0]
        if normalize_label(bottom['label']) == base:
            continue

        step = float(np.median(group['gaps'])) if group['gaps'] else bottom['h']
        cur = bottom
        for _ in range(max_cubes - len(group['members'])):
            cy = cur['cy'] + step
            y1 = int(cy - cur['h'] / 2)
            y2 = min(h_img, int(cy + cur['h'] / 2))
            box = (cur['x1'], y1, cur['x2'], y2)
            tag = f"below {cur['label']} at ({cur['cx']:.0f},{cur['cy']:.0f})"

            if y2 - y1 < 0.5 * cur['h']:          # mostly off the bottom of frame
                PROBE_LOG.append(f"{tag}: off frame")
                break

            owner = next((c for c in detected if c not in group['members']
                          and _overlap_frac(box, c) > PROBE_MAX_OVERLAP), None)
            if owner is not None:
                PROBE_LOG.append(f"{tag}: skipped, region belongs to detected "
                                 f"{owner['label']}")
                break

            color, cells = classify_patch(image, *box)
            if color is None or color not in names_by_color:
                best = max(cells, key=lambda k: sum(cells[k])) if cells else None
                detail = (f" (closest {best}: cells "
                          f"{[round(v, 2) for v in cells[best]]})") if best else ""
                PROBE_LOG.append(f"{tag}: no cube{detail}")
                break

            PROBE_LOG.append(f"{tag}: inferred {color}, cells "
                             f"{[round(v, 2) for v in cells[color]]}")

            cube = {
                'label': names_by_color[color],
                'yolo_label': None,
                'recoloured': False,
                'inferred': True,
                'reject': False,
                'conf': 0.0,
                'x1': cur['x1'], 'y1': y1, 'x2': cur['x2'], 'y2': y2,
                'cx': cur['cx'], 'cy': (y1 + y2) / 2.0,
                'w': cur['w'], 'h': float(y2 - y1),
                'angle': 0.0,
                'hsv': roi_hsv(image, cur['x1'], y1, cur['x2'], y2),
            }
            inferred.append(cube)
            cur = cube
            if color == base:
                break

    return inferred


def same_cube(a, b):
    """True if two boxes are faces / duplicates of the same physical cube."""
    ix = min(a['x2'], b['x2']) - max(a['x1'], b['x1'])      # < 0 means a gap
    iy = min(a['y2'], b['y2']) - max(a['y1'], b['y1'])
    if iy <= 0:
        return False
    small = min(a['w'] * a['h'], b['w'] * b['h'])
    if ix > 0 and ix * iy >= MERGE_CONTAIN * small:          # one inside the other
        return True
    if normalize_label(a['label']) != normalize_label(b['label']):
        return False
    return (iy >= MERGE_Y_OVERLAP * min(a['h'], b['h'])
            and -ix <= MERGE_X_GAP * max(a['w'], b['w']))    # side by side


def merge_duplicates(cubes):
    """Drop boxes that are another face / duplicate of a more confident box."""
    kept = []
    for c in sorted(cubes, key=lambda c: -c['conf']):
        twin = next((k for k in kept if same_cube(c, k)), None)
        if twin is None:
            kept.append(c)
        else:
            twin.setdefault('merged', []).append(
                f"{normalize_label(c['label'])} {c['conf']:.2f}")
    return kept


def detect_and_group(model, image):
    """Detection -> duplicate merge -> grouping -> hidden-cube inference -> regrouping."""
    cubes = merge_duplicates(detect_cubes(model, image))
    hidden = infer_hidden_cubes(image, group_cubes(cubes),
                                class_names_by_color(model), detected=cubes)
    cubes = cubes + hidden
    return cubes, group_cubes(cubes)


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def normalize_label(label):
    """'red_cube' -> 'red', 'Green Block' -> 'green'."""
    token = str(label).lower().replace('-', ' ').replace('_', ' ')
    parts = [p for p in token.split() if p not in LABEL_NOISE_TOKENS]
    return ' '.join(parts) if parts else token.strip()


def _cluster_width(cluster):
    """Typical cube width in a column; narrow side-face boxes are ignored."""
    full = [m['w'] for m in cluster if not is_side_face(m)]
    return float(np.median(full or [m['w'] for m in cluster]))


def is_side_face(c):
    return c['h'] > 0 and c['w'] / c['h'] < SIDE_FACE_ASPECT


def _can_join(cube, cluster):
    """Same horizontal column AND similar apparent size."""
    ref_w = _cluster_width(cluster)
    if ref_w <= 0:
        return False, None

    centre = float(np.mean([m['cx'] for m in cluster]))
    dist = abs(cube['cx'] - centre)

    # A narrow side-face box (cube turned, only its side detected) has a
    # different width and an offset centre: join if it touches the column.
    if is_side_face(cube) or all(is_side_face(m) for m in cluster):
        x1 = min(m['x1'] for m in cluster)
        x2 = max(m['x2'] for m in cluster)
        gap = max(x1 - cube['x2'], cube['x1'] - x2)          # < 0: overlap
        wide = max(cube['w'], ref_w)
        return gap <= SIDE_FACE_X_GAP * wide, dist

    if abs(cube['w'] - ref_w) / ref_w > SIZE_RATIO_TOL:
        return False, None

    x_tol = SAME_STACK_X_FRAC * 0.5 * (cube['w'] + ref_w)
    return dist < x_tol, dist


def group_cubes(cubes):
    """Cluster cubes into vertical columns, ordered left to right."""
    if not cubes:
        return []

    clusters = []
    for cube in sorted(cubes, key=lambda c: c['cx']):
        best, best_d = None, None
        for cluster in clusters:
            ok, dist = _can_join(cube, cluster)
            if ok and (best_d is None or dist < best_d):
                best, best_d = cluster, dist
        if best is None:
            clusters.append([cube])
        else:
            best.append(cube)

    groups = []
    for cluster in clusters:
        # Larger cy means lower in the image, i.e. nearer the bottom of a stack.
        cluster.sort(key=lambda c: -c['cy'])
        for pos, cube in enumerate(cluster):
            cube['stack_pos'] = pos

        gaps = [cluster[i]['cy'] - cluster[i + 1]['cy'] for i in range(len(cluster) - 1)]
        xs = [c['cx'] for c in cluster]
        spread = max(xs) - min(xs)
        ref_w = _cluster_width(cluster)

        groups.append({
            'members': cluster,
            'is_stack': len(cluster) >= MIN_STACK_SIZE,
            'order': [c['label'] for c in cluster],
            'gaps': [round(g, 1) for g in gaps],
            'big_gaps': [],
            'gaps_ok': True,
            'x_spread': round(spread, 1),
            'x_spread_ok': spread <= X_SPREAD_WARN_FRAC * ref_w,
            'typical_gap': None,
        })

    all_gaps = [gap for grp in groups for gap in grp['gaps']]
    if len(all_gaps) >= 2:
        g_med = float(np.median(all_gaps))
        for grp in groups:
            grp['typical_gap'] = round(g_med, 1)
            grp['big_gaps'] = [g for g in grp['gaps'] if g > GAP_MULTIPLE_WARN * g_med]
            grp['gaps_ok'] = not grp['big_gaps']

    groups.sort(key=lambda g: np.mean([m['cx'] for m in g['members']]))
    return groups


# --------------------------------------------------------------------------
# Order check (pure functions - used on YOLO's, the VLM's and the final reading)
# --------------------------------------------------------------------------
# A "state" is {'stacks': [[colour, ...] bottom -> top, ...], 'table': [colour, ...]}.


def goal_colours():
    return [normalize_label(c) for c in EXPECTED_ORDER]


def detector_state(groups):
    """YOLO's reading as a state."""
    return {'stacks': [[normalize_label(c) for c in g['order']]
                       for g in groups if g['is_stack']],
            'table': [normalize_label(g['order'][0])
                      for g in groups if not g['is_stack']]}


def all_colours(state):
    return [c for s in state['stacks'] for c in s] + list(state['table'])


def same_state(a, b):
    return (sorted(map(tuple, a['stacks'])) == sorted(map(tuple, b['stacks']))
            and sorted(a['table']) == sorted(b['table']))


def state_summary(state):
    """'red>green | black | white' (stacks bottom > top, then single cubes)."""
    parts = ['>'.join(s) for s in state['stacks']] + list(state['table'])
    return ' | '.join(parts) if parts else 'no cubes'


def evaluate_state(state):
    """
    Order check against EXPECTED_ORDER:
      complete    - one stack is exactly the goal
      wrong       - some stack of 2+ cubes has a cube in the wrong place,
                    counting from the bottom
      in_progress - otherwise (nothing stacked wrongly, cubes only missing)
    plus the exact anomalies and goal colours not visible at all.
    """
    goal = goal_colours()
    piles = state['stacks'] + [[c] for c in state['table']]

    left = Counter(all_colours(state))
    missing = []
    for c in goal:
        if left[c] > 0:
            left[c] -= 1
        else:
            missing.append(c)

    anomalies = []
    for s in state['stacks']:
        for i, c in enumerate(s):
            if i >= len(goal):
                anomalies.append(f"extra {c} cube on top of the stack {'>'.join(s)}")
                break
            if c != goal[i]:
                anomalies.append(f"in the stack {'>'.join(s)}, cube {i + 1} from the "
                                 f"bottom is {c} but should be {goal[i]}")
                break

    if any(p == goal for p in piles):
        status = 'complete'
    elif anomalies:
        status = 'wrong'
    else:
        status = 'in_progress'
    return {'status': status, 'anomalies': anomalies, 'missing': missing}


def problem_text(verdict, state):
    """One readable sentence for the worker, from a code verdict."""
    goal = goal_colours()
    if verdict['status'] == 'complete':
        return "All cubes are in the right order. Well done."
    if verdict['anomalies']:
        return "; ".join(a[0].upper() + a[1:] for a in verdict['anomalies']) + "."
    piles = state['stacks'] + [[c] for c in state['table']]
    k = max((len(p) for p in piles if p == goal[:len(p)]), default=0)
    done = ' > '.join(goal[:k]) if k else 'nothing yet'
    return f"Correct so far: {done}. Still to add: {', '.join(goal[k:])}."


def beneath_map(state):
    """{colour: what it stands on} - compares readings without stack numbering."""
    m = {}
    for s in state['stacks']:
        for i, c in enumerate(s):
            m[c] = s[i - 1] if i else 'table'
    for c in state['table']:
        m[c] = 'table'
    return m


def compare_readings(det, vlm):
    """Where YOLO and the VLM disagree, and whether the VLM confirms YOLO's anomaly."""
    d, v = Counter(all_colours(det)), Counter(all_colours(vlm))
    bd, bv = beneath_map(det), beneath_map(vlm)
    dv, vv = evaluate_state(det), evaluate_state(vlm)
    return {
        'same_reading': same_state(det, vlm),
        'missed_by_detector': list((v - d).elements()),
        'missed_by_vlm': list((d - v).elements()),
        'placed_differently': [{'cube': c, 'detector_on': bd[c], 'vlm_on': bv[c]}
                               for c in bd if c in bv and bd[c] != bv[c]],
        'detector_status': dv['status'],
        'vlm_status': vv['status'],
        'status_agrees': dv['status'] == vv['status'],
        # None when YOLO saw no anomaly; else True if the VLM sees the same one.
        'detector_anomaly_confirmed': (None if dv['status'] == 'complete' else
                                       dv['status'] == vv['status']
                                       and dv['anomalies'] == vv['anomalies']),
    }


# --------------------------------------------------------------------------
# Display
# --------------------------------------------------------------------------

FONT = cv2.FONT_HERSHEY_SIMPLEX
COLOR_OK = (60, 170, 60)          # BGR
COLOR_BAD = (50, 50, 215)
COLOR_WARN = (0, 150, 235)
COLOR_IDLE = (110, 110, 110)
COLOR_NOTE = (230, 170, 255)      # VLM / detector comparison lines
PANEL_BG = (25, 25, 25)
PANEL_ALPHA = 0.78
TEXT_MAIN = (245, 245, 245)
TEXT_DIM = (150, 150, 150)

# status -> panel header
STATUS_STYLE = {
    'complete':    ("STACK COMPLETE", COLOR_OK),
    'in_progress': ("KEEP GOING", COLOR_WARN),
    'wrong':       ("FIX THE STACK", COLOR_BAD),
}


def cube_color(label):
    """BGR colour matching the cube's class, e.g. 'red_cube' -> red."""
    name = normalize_label(label)
    if name in CUBE_COLORS:
        return CUBE_COLORS[name]
    for token in name.split():               # 'dark red' -> 'red'
        if token in CUBE_COLORS:
            return CUBE_COLORS[token]
    return FALLBACK_COLORS[zlib.crc32(name.encode()) % len(FALLBACK_COLORS)]


def contrast_color(bgr):
    """Black or white, whichever reads better on top of the given colour."""
    b, g, r = bgr
    luminance = 0.114 * b + 0.587 * g + 0.299 * r
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def dashed_rect(image, p1, p2, color, thickness=2, dash=12):
    """Rectangle drawn as dashes (OpenCV has no built-in dashed line)."""
    (x1, y1), (x2, y2) = p1, p2
    for x in range(x1, x2, dash * 2):
        cv2.line(image, (x, y1), (min(x + dash, x2), y1), color, thickness)
        cv2.line(image, (x, y2), (min(x + dash, x2), y2), color, thickness)
    for y in range(y1, y2, dash * 2):
        cv2.line(image, (x1, y), (x1, min(y + dash, y2)), color, thickness)
        cv2.line(image, (x2, y), (x2, min(y + dash, y2)), color, thickness)


def wrap_text(text, max_width, scale, thickness):
    """Split text into lines that fit max_width pixels."""
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, FONT, scale, thickness)[0][0] <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def group_tags(groups):
    """{id(group): 'S1'} for stacks, numbered left to right. Singles get no tag."""
    tags, n = {}, 0
    for g in groups:
        if g['is_stack']:
            n += 1
            tags[id(g)] = f"S{n}"
    return tags


def cube_tag(cube, group, tags):
    """'S1-2 green' for a stacked cube (2nd from the bottom), 'green' for a single."""
    colour = normalize_label(cube['label'])
    tag = tags.get(id(group))
    return f"{tag}-{cube['stack_pos'] + 1} {colour}" if tag else colour


def draw_label(image, text, box, bg, fg, scale, thick):
    """
    Filled tag beside the box, vertically centred: to the right, or to the
    left if there is no room. Never above, so in a stack it does not cover
    the cube sitting on top.
    """
    x1, y1, x2, y2 = box
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    pad = max(2, int(4 * scale))
    bw, bh = tw + 2 * pad, th + base + 2 * pad
    lx = x2 + 4 if x2 + 4 + bw <= image.shape[1] else max(0, x1 - 4 - bw)
    ly = int((y1 + y2) / 2 - bh / 2)
    ly = max(0, min(ly, image.shape[0] - bh))
    cv2.rectangle(image, (lx, ly), (lx + bw, ly + bh), bg, -1)
    cv2.rectangle(image, (lx, ly), (lx + bw, ly + bh), fg, 1)
    cv2.putText(image, text, (lx + pad, ly + bh - base - pad), FONT, scale, fg,
                thick, cv2.LINE_AA)


def draw_detections(image, groups, offset=(0, 0), scale=1.0,
                    font_scale=0.45, thickness=2):
    """
    Detections only: a box in each cube's own colour and a compact tag.
    No verdicts, flags or guidance. `offset` / `scale` map frame coordinates
    onto a cropped / resized copy (used for the image sent to the VLM).
    """
    ox, oy = offset
    h, w = image.shape[:2]
    tags = group_tags(groups)
    for g in groups:
        for c in g['members']:
            x1, y1 = int((c['x1'] - ox) * scale), int((c['y1'] - oy) * scale)
            x2, y2 = int((c['x2'] - ox) * scale), int((c['y2'] - oy) * scale)
            if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
                continue
            colour = cube_color(c['label'])
            edge = contrast_color(colour)
            if c.get('inferred'):
                dashed_rect(image, (x1, y1), (x2, y2), edge, thickness + 2)
                dashed_rect(image, (x1, y1), (x2, y2), colour, thickness)
            else:
                cv2.rectangle(image, (x1, y1), (x2, y2), edge, thickness + 2)
                cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness)
            draw_label(image, cube_tag(c, g, tags) + (" (geometric reasoning)" if c.get("inferred") else ""), (x1, y1, x2, y2), colour, edge,
                       font_scale, max(1, thickness - 1))
    return image


def blend_rect(image, x1, y1, x2, y2, colour, alpha):
    """Semi-transparent filled rectangle."""
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    fill = np.full_like(roi, colour)
    image[y1:y2, x1:x2] = cv2.addWeighted(fill, alpha, roi, 1 - alpha, 0)


def wrap_hanging(prefix, text, max_width, scale, thickness):
    """Wrap 'prefix + text' so continuation lines align under the text."""
    indent = cv2.getTextSize(prefix, FONT, scale, thickness)[0][0]
    lines = wrap_text(text, max_width - indent, scale, thickness) or [""]
    return [(prefix if i == 0 else "", indent, line) for i, line in enumerate(lines)]


def choose_panel_side(groups, width):
    """Put the panel on the side away from the cubes."""
    if not groups:
        return 'right'
    cx = np.mean([m['cx'] for g in groups for m in g['members']])
    return 'left' if cx > width / 2 else 'right'


def draw_instruction_panel(image, header, header_colour, problem, steps,
                           footer, footer_colour, dim=False, side='right',
                           notes=()):
    """
    Panel for the worker: coloured header, one-line problem, optional notes
    (what was seen / where YOLO and the VLM disagree), EVERY step numbered,
    and a status footer. Steps are greyed out (dim) while a new placement is
    being checked, so the worker can still read them.
    """
    h, w = image.shape[:2]
    margin, pad = 12, 14
    pw = int(min(max(380, 0.36 * w), 560, w - 2 * margin))
    x1 = w - pw - margin if side == 'right' else margin
    inner = pw - 2 * pad

    s_head, s_prob, s_step, s_foot, s_note = 0.72, 0.52, 0.6, 0.48, 0.45
    gap_step = 8

    def lh(scale):
        return int(cv2.getTextSize("Ag", FONT, scale, 1)[0][1] * 1.9)

    prob_lines = wrap_text(problem, inner, s_prob, 1) if problem else []
    note_lines = [ln for n in notes for ln in wrap_text(n, inner, s_note, 1)]
    step_blocks = [wrap_hanging(f"{i}. ", s, inner, s_step, 1)
                   for i, s in enumerate(steps, 1)]

    head_h = lh(s_head) + 10
    body = (len(prob_lines) * lh(s_prob) + (10 if prob_lines else 0)
            + len(note_lines) * lh(s_note) + (10 if note_lines else 0)
            + sum(len(b) * lh(s_step) + gap_step for b in step_blocks))
    foot_h = lh(s_foot) + 8
    max_body = h - 2 * margin - head_h - foot_h - 2 * pad
    while step_blocks and body > max_body:          # very long answer: trim
        body -= len(step_blocks[-1]) * lh(s_step) + gap_step
        step_blocks.pop()
    total = head_h + pad + body + foot_h + pad

    y = margin
    blend_rect(image, x1, y, x1 + pw, y + total, PANEL_BG, PANEL_ALPHA)
    cv2.rectangle(image, (x1, y), (x1 + pw, y + head_h), header_colour, -1)
    cv2.putText(image, header, (x1 + pad, y + head_h - 12), FONT, s_head,
                (255, 255, 255), 2, cv2.LINE_AA)
    y += head_h + pad

    for line in prob_lines:
        y += lh(s_prob)
        cv2.putText(image, line, (x1 + pad, y - 6), FONT, s_prob,
                    TEXT_DIM if dim else (210, 210, 210), 1, cv2.LINE_AA)
    if prob_lines:
        y += 10

    for line in note_lines:
        y += lh(s_note)
        cv2.putText(image, line, (x1 + pad, y - 6), FONT, s_note,
                    TEXT_DIM if dim else COLOR_NOTE, 1, cv2.LINE_AA)
    if note_lines:
        y += 10

    colour = TEXT_DIM if dim else TEXT_MAIN
    for block in step_blocks:
        for prefix, indent, line in block:
            y += lh(s_step)
            if prefix:
                cv2.putText(image, prefix, (x1 + pad, y - 6), FONT, s_step,
                            colour, 2, cv2.LINE_AA)
            cv2.putText(image, line, (x1 + pad + indent, y - 6), FONT, s_step,
                        colour, 1, cv2.LINE_AA)
        y += gap_step
    if len(step_blocks) < len(steps):
        cv2.putText(image, f"(+{len(steps) - len(step_blocks)} more - see terminal)",
                    (x1 + pad, y + 4), FONT, 0.45, TEXT_DIM, 1, cv2.LINE_AA)

    fy = margin + total - pad // 2 - 6
    cv2.line(image, (x1 + pad, fy - lh(s_foot)), (x1 + pw - pad, fy - lh(s_foot)),
             (70, 70, 70), 1)
    cv2.circle(image, (x1 + pad + 6, fy - 6), 6, footer_colour, -1, cv2.LINE_AA)
    cv2.putText(image, footer, (x1 + pad + 20, fy), FONT, s_foot, (200, 200, 200),
                1, cv2.LINE_AA)


def result_notes(res):
    """Short lines for the panel: final reading and YOLO/VLM disagreements."""
    notes = [f"Goal (bottom>top): {'>'.join(EXPECTED_ORDER)}",
             f"Seen (bottom>top): {state_summary(res['state'])}  [{res['state_source']}]"]
    if not res.get('vlm_called', True):
        notes.append(f"VLM not called: {res['why_no_vlm']}")
    elif res.get('triggers'):
        notes.append(f"VLM called: {res['triggers'][0]}")
    if res.get('look_problems'):
        if res.get('vlm_state') is not None:
            notes.append(f"VLM said: {state_summary(res['vlm_state'])}")
        notes.append(f"Rejected: {'; '.join(res['look_problems'])}")
    cmp = res.get('comparison') or {}
    if cmp.get('missed_by_detector'):
        notes.append(f"VLM found {', '.join(cmp['missed_by_detector'])} - missed by YOLO")
    if cmp.get('missed_by_vlm') and res.get('look_problems'):
        # (when the VLM reading is used, ignored YOLO boxes are in the JSON only)
        notes.append(f"VLM did not see {', '.join(cmp['missed_by_vlm'])}")
    if cmp.get('placed_differently'):
        notes.append("YOLO and VLM disagree on where: "
                     + ", ".join(m['cube'] for m in cmp['placed_differently']))
    return notes


def instructions_from_result(res):
    """(header, colour, problem, steps, notes) for a finished VLM result."""
    notes = result_notes(res)
    if res['error']:
        return "VLM ERROR", COLOR_BAD, res['error'], [], notes
    verdict = res['verdict']
    if not res['parsed']:
        return ("NO INSTRUCTIONS", STATUS_STYLE.get(verdict['status'], (0, COLOR_WARN))[1],
                problem_text(verdict, res['state'])
                + " The planner gave no usable moves - press v to ask again.",
                [], notes)
    header, colour = STATUS_STYLE.get(res['status'], ("CHECK THE STACK", COLOR_WARN))
    problem = res['problem'] or problem_text(verdict, res['state'])
    if res['status'] == 'complete':
        problem = problem_text(verdict, res['state'])
    if not res['verified']:
        # Shown, but clearly marked: the moves did not survive the replay check.
        header, colour = "CHECK THESE STEPS", COLOR_WARN
        problem = f"{problem} (Plan could not be verified - follow with care.)".strip()
    return header, colour, problem, res['steps'], notes


def cube_line(cube):
    """One report line for a cube, including colour-check details."""
    hsv = cube.get('hsv')
    hsv_txt = f" hsv=({hsv[0]:.0f},{hsv[1]:.0f},{hsv[2]:.0f})" if hsv else ""
    fix = f" [YOLO said {cube['yolo_label']}]" if cube.get('recoloured') else ""
    if cube.get('inferred'):
        fix = " [INFERRED from colour below stack - not detected by YOLO]"
    return (f"{cube['label']:<12} conf={cube['conf']:.2f} "
            f"centre=({cube['cx']:.0f},{cube['cy']:.0f}) "
            f"box={cube['w']:.0f}x{cube['h']:.0f}{hsv_txt}{fix}")


def short_summary(groups):
    """Compact detection-only description, e.g. 'S1[red>black] | green | white'."""
    tags = group_tags(groups)
    parts = []
    for g in groups:
        order = [normalize_label(c) for c in g['order']]
        parts.append(f"{tags[id(g)]}[{'>'.join(order)}]" if g['is_stack'] else order[0])
    return ' | '.join(parts) if parts else 'no cubes'


def build_report(image_name, groups):
    """Detections only - what was seen, with detector-quality warnings."""
    tags = group_tags(groups)
    stacks = [g for g in groups if g['is_stack']]
    singles = [g for g in groups if not g['is_stack']]
    lines = [f"Image: {image_name}",
             f"Goal (bottom -> top): {' -> '.join(EXPECTED_ORDER)}",
             f"Stacks detected: {len(stacks)}",
             f"Single cubes on table: {len(singles)}", ""]
    for g in stacks:
        lines.append(f"Stack {tags[id(g)]}: {len(g['members'])} cubes (bottom -> top)")
        for cube in g['members']:
            lines.append(f"  {cube_tag(cube, g, tags):<14} {cube_line(cube)}")
        lines.append(f"  gaps   : {g['gaps']} (typical {g['typical_gap']}), "
                     f"x-spread {g['x_spread']}")
        if not g['gaps_ok']:
            lines.append(f"  detector warning: oversized gap(s) {g['big_gaps']} - a cube "
                         f"may be hidden or missed")
        if not g['x_spread_ok']:
            lines.append(f"  detector warning: members span {g['x_spread']}px - may be "
                         f"two adjacent stacks read as one")
        lines.append("")
    if singles:
        lines.append("Single cubes:")
        lines += [f"  {cube_line(g['members'][0])}" for g in singles]
        lines.append("")
    return "\n".join(lines)


def print_debug(cubes, model=None, frame=None):
    """Dump every detection with HSV, plus near-misses below CONF."""
    print(f"--- {len(cubes)} cube(s) | ACHROMATIC_S_MAX={ACHROMATIC_S_MAX} "
          f"WHITE_V_MIN={WHITE_V_MIN} BLACK_V_MAX={BLACK_V_MAX} ---")
    for c in sorted(cubes, key=lambda c: -c['cy']):
        print(f"  {cube_line(c)}")

    if model is not None and frame is not None:
        raw = detect_cubes(model, frame, conf=DEBUG_CONF, checks=False)
        near = [c for c in raw if c['conf'] < CONF]
        print(f"  near-misses between conf={DEBUG_CONF} and {CONF}: {len(near)}")
        for c in sorted(near, key=lambda c: -c['conf']):
            print(f"    {cube_line(c)}")
        if not near:
            print("    none - anything missing was not seen by YOLO at all")

    print(f"  hidden-cube probes ({len(PROBE_LOG)}):")
    for line in PROBE_LOG:
        print(f"    {line}")


# --------------------------------------------------------------------------
# Cameras
# --------------------------------------------------------------------------


class OakCamera:
    """RGB stream from a Luxonis OAK-D Pro (DepthAI v3)."""

    def __init__(self, size, fps):
        import depthai as dai
        self.pipeline = dai.Pipeline()
        cam = self.pipeline.create(dai.node.Camera).build(
            dai.CameraBoardSocket.CAM_A, sensorFps=fps)

        if MANUAL_WB_K is not None:
            try:
                cam.initialControl.setManualWhiteBalance(int(MANUAL_WB_K))
                print(f"OAK white balance locked at {MANUAL_WB_K} K")
            except Exception as exc:          # API differs across DepthAI versions
                print(f"WARNING: could not lock white balance ({exc}); using auto")

        # maxSize=1, non-blocking -> always process the newest frame
        self.queue = cam.requestOutput(size).createOutputQueue(maxSize=1, blocking=False)
        self.pipeline.start()
        print("OAK camera started:", self.pipeline.getDefaultDevice().getDeviceName())

    def read(self):
        msg = self.queue.get()
        return None if msg is None else msg.getCvFrame()

    def release(self):
        self.pipeline.stop()


class WebcamCamera:
    """Any camera OpenCV can open (USB webcam, RealSense RGB via V4L2, ...)."""

    def __init__(self, index, size, fps):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            sys.exit(f"Could not open camera index {index}")
        if size is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        ok, frame = self.cap.read()
        return frame if ok else None

    def release(self):
        self.cap.release()


# --------------------------------------------------------------------------
# VLM reasoning (Ollama)
# --------------------------------------------------------------------------
#
#   1. LOOK  (vision model, image + YOLO boxes): what stacks and single cubes
#      are there, bottom -> top? Complements YOLO. Re-asked with the reason if
#      it drops a confident YOLO cube or lists a colour twice.
#   2. Code checks the order of the accepted reading (evaluate_state).
#   3. PLAN  (text model): status, problem and every move as {cube, to}.
#      Code replays the moves; impossible / unfinished plans and a status that
#      contradicts the order check are sent back with the reason.
# Answer formats are enforced with Ollama JSON schemas.

PLAN_STATUSES = ('complete', 'in_progress', 'wrong')


def known_colours(groups=()):
    """Colour words the VLM may use: the goal colours + anything detected."""
    cols = [normalize_label(c) for c in EXPECTED_ORDER] + list(MODEL_COLOURS)
    cols += [normalize_label(c['label']) for g in groups for c in g['members']]
    return list(dict.fromkeys(cols))


def to_colour(text, colours):
    """'S1-2 black cube' -> 'black'; None if no known colour is named."""
    words = normalize_label(str(text)).replace(',', ' ').split()
    for w in words:
        if w in colours:
            return w
    return None


def describe_detections(groups):
    """YOLO's findings as plain facts for the LOOK prompt - no verdicts."""
    tags = group_tags(groups)
    lines, singles = [], []
    for g in groups:
        if g['is_stack']:
            cubes = ", ".join(normalize_label(c['label'])
                              + (" (guessed)" if c.get('inferred') else "")
                              for c in g['members'])
            note = "" if g['gaps_ok'] else "  [gap: a cube may be hidden or missed here]"
            lines.append(f"- stack {tags[id(g)]}, bottom to top: {cubes}{note}")
        else:
            singles.append(normalize_label(g['order'][0]))
    if singles:
        lines.append(f"- single cubes on the table: {', '.join(singles)}")
    return "\n".join(lines) if lines else "- no cubes detected"


def look_prompt(groups, colours):
    unique = ("\nThere is only ONE cube of each colour, so list every colour at most once."
              if UNIQUE_COLOURS else "")
    if not VLM_ANNOTATE:
        # Independent reading from the raw photo - no detector hints that could
        # bias it (e.g. copying a wrong split into two stacks).
        return f"""The photo shows coloured cubes on a table, seen from the front by a fixed camera.
Some cubes may be stacked on top of each other, others stand alone on the table.
A cube may be turned slightly, so you can see two of its sides - it is still ONE cube.
Cubes stacked on top of each other form ONE stack, even if they are not perfectly aligned.
Look carefully from the table upward and report what is on the table:
"stacks" has one list for every stack of two or more cubes, with the colours from the BOTTOM cube (touching the table) up to the TOP cube.
"table" has the colours of the single cubes standing directly on the table.{unique}
Use only these colour words: {', '.join(colours)}."""
    return f"""The photo shows coloured cubes on a table, seen from the front by a fixed camera.
An object detector has drawn a box around each cube, in the cube's own colour, with a tag beside it.
A tag like "S1-2 green" means: stack S1 (stacks are numbered from left to right), the 2nd cube counted from the bottom.
A tag with only a colour means a single cube standing directly on the table.
A dashed box is a cube the detector only guessed.

The detector's reading:
{describe_detections(groups)}

Check this reading against the photo. The detector often MISSES cubes, especially the
bottom cube of a stack and cubes without a box. Boxes can also be doubled or have the wrong colour.
Then report what is really on the table:
"stacks" has one list for every stack of two or more cubes, with the colours from the BOTTOM cube up to the TOP cube.
"table" has the colours of the single cubes standing directly on the table.{unique}
Use only these colour words: {', '.join(colours)}."""


def look_problems(seen, groups):
    """Reasons a LOOK reading cannot be accepted (empty list = accept)."""
    probs = []
    counts = Counter(all_colours(seen))
    if not counts and groups:
        probs.append("your reading is empty, but there are cubes in the photo")
    if UNIQUE_COLOURS:
        for c, n in counts.items():
            if n > 1:
                probs.append(f"you listed the {c} cube {n} times, but there is only "
                             f"one {c} cube")
    sure = Counter(normalize_label(c['label']) for g in groups for c in g['members']
                   if not c.get('inferred') and c['conf'] >= VLM_TRUST_CONF)
    for c in sorted(set((sure - counts).elements())):
        probs.append(f"a detector is confident there is a {c} cube in the photo, "
                     f"but your reading has no {c} cube - look again")
    return probs


def state_text(stacks, table):
    lines = [f"- stack {chr(65 + i)}, from bottom to top: {', '.join(s)}"
             for i, s in enumerate(stacks)]
    lines += [f"- a single {c} cube standing on the table" for c in table]
    return "\n".join(lines) if lines else "- no cubes"


def plan_prompt(stacks, table, colours):
    goal = goal_colours()
    return f"""A worker must build ONE stack of {len(goal)} cubes.
GOAL, from the bottom cube (on the table) to the top cube: {', '.join(goal)}.

What is on the table now:
{state_text(stacks, table)}

Rules:
- Only the top cube of a stack, or a single cube on the table, can be picked up.
- Each move picks up ONE cube and puts it either on the table or on top of another cube.
- Do not move cubes that are already in the right place from the bottom of the goal stack upward.

Answer with:
"status": "complete" if one stack is already exactly {', '.join(goal)};
          "wrong" if any stack of two or more cubes has a cube in the wrong place, counting from the bottom ({goal[0]} must be at the bottom);
          "in_progress" otherwise (nothing is stacked wrongly, cubes are only missing from the goal stack).
"problem": one sentence that names the colours involved, saying what is wrong or what is missing.
"moves": EVERY move, in order, until the goal stack is finished. Each move has "cube" = the colour of the cube to pick up, and "to" = "table" or the colour of the cube to put it on. Use an empty list only when the status is complete.

How to plan: keep the cubes that are already correct from the bottom up; take every
wrong cube above them down to the table, starting from the top; then add the missing
goal cubes one by one, each on top of the previous goal cube.

{plan_example(goal)}
Colour words: {', '.join(colours)}."""


def plan_example(goal):
    """Worked example for the plan prompt, built from the real goal colours."""
    if len(goal) < 4:
        return ""
    a, b, c, d = goal[:4]
    return f"""Example: stack {a}, {b}, {d} and a single {c} cube on the table.
Answer: {{"status": "wrong", "problem": "The {d} cube is on {b}, but {c} must come next.",
 "moves": [{{"cube": "{d}", "to": "table"}}, {{"cube": "{c}", "to": "{b}"}},
           {{"cube": "{d}", "to": "{c}"}}]}}
"""


def look_schema(colours):
    col = {'type': 'string', 'enum': colours}
    return {'type': 'object',
            'properties': {'stacks': {'type': 'array',
                                      'items': {'type': 'array', 'items': col}},
                           'table': {'type': 'array', 'items': col}},
            'required': ['stacks', 'table']}


def plan_schema(colours):
    return {'type': 'object',
            'properties': {
                'status': {'type': 'string', 'enum': list(PLAN_STATUSES)},
                'problem': {'type': 'string'},
                'moves': {'type': 'array', 'items': {
                    'type': 'object',
                    'properties': {'cube': {'type': 'string', 'enum': colours},
                                   'to': {'type': 'string',
                                          'enum': colours + ['table']}},
                    'required': ['cube', 'to']}}},
            'required': ['status', 'problem', 'moves']}


def prepare_vlm_image(frame, groups, crop=None):
    """
    The image the VLM sees: the settled frame (cropped to the work area if
    set), scaled so its longest side is VLM_MAX_SIDE, with the YOLO boxes and
    tags drawn on it. Returns JPEG bytes.
    """
    ox, oy = 0, 0
    img = frame
    if crop is not None and VLM_CROP_TO_WORK_AREA:
        x1, y1, x2, y2 = crop
        if x2 - x1 > 32 and y2 - y1 > 32:
            img, ox, oy = frame[y1:y2, x1:x2], x1, y1
    h, w = img.shape[:2]
    scale = min(VLM_MAX_SIDE / max(h, w), 2.0)       # up to 2x for small crops
    img = cv2.resize(img, (int(w * scale), int(h * scale)),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    if VLM_ANNOTATE:
        draw_detections(img, groups, offset=(ox, oy), scale=scale,
                        font_scale=0.6, thickness=2)
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("could not encode frame for the VLM")
    return buf.tobytes()


def resolve_model(want, names):
    """
    Installed Ollama name for `want`, or None.
    'qwen3-vl' matches 'qwen3-vl:latest'; if the exact tag is missing, an
    installed tag of the same family is used ('qwen3-vl:8b' -> 'qwen3-vl:8b-instruct',
    'qwen3-vl:latest', ...), preferring one whose tag starts with the wanted tag.
    """
    family, _, tag = want.lower().partition(':')    # names are matched ignoring case
    full = f"{family}:{tag or 'latest'}"
    for n in names:
        if n.lower() in (want.lower(), full):
            return n
    same = [n for n in names if n.lower().split(':')[0] == family]
    pref = [n for n in same if tag and n.lower().split(':', 1)[-1].startswith(tag)]
    return (pref or same or [None])[0]


class OllamaVLM:
    """Talks to a local Ollama server (REST) or the `ollama` command (CLI)."""

    def __init__(self, model=VLM_MODEL, backend=VLM_BACKEND, url=OLLAMA_URL,
                 timeout=VLM_TIMEOUT, planner=None):
        self.model, self.backend = model, backend
        self.planner = planner or model          # text model for the PLAN step
        self.url, self.timeout = url.rstrip('/'), timeout
        self.schemas_ok = True                   # Ollama >= 0.5 accepts schemas
        self.think_ok = True                     # send think=false (Ollama >= 0.9)

    def _post(self, path, payload, timeout):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(payload).encode(),
            headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())

    def _capabilities(self, model):
        """Model capabilities from /api/show (newer Ollama), or None if unknown."""
        try:
            caps = self._post('/api/show', {'model': model}, 10).get('capabilities')
            return list(caps) if caps else None
        except Exception:
            return None

    def check(self):
        """(ok, message). Run once at start-up."""
        if self.backend == 'cli':
            if shutil.which('ollama') is None:
                return False, "`ollama` not found on PATH"
            out = subprocess.run(['ollama', 'list'], capture_output=True, text=True).stdout
            names = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
        else:
            try:
                with urllib.request.urlopen(self.url + '/api/tags', timeout=5) as r:
                    tags = json.loads(r.read().decode())
            except (urllib.error.URLError, OSError) as exc:
                return False, f"Ollama not reachable at {self.url} ({exc}) - run `ollama serve`"
            names = [m.get('name') or m.get('model', '') for m in tags.get('models', [])]

        notes = []
        for attr in ('model', 'planner'):
            want = getattr(self, attr)
            got = resolve_model(want, names)
            if got is None:
                return False, (f"model '{want}' not found on the Ollama server at "
                               f"{self.url}. Installed: {', '.join(names) or 'none'}. "
                               f"Pass one with --vlm-model, e.g. --vlm-model {names[0]}"
                               if names else
                               f"no models on the Ollama server at {self.url} - is "
                               f"`ollama pull` using the same server (OLLAMA_HOST)?")
            if got != want:
                notes.append(f"'{want}' not installed, using '{got}'")
                setattr(self, attr, got)
        for n in notes:
            print(f"VLM: {n}")
        if self.backend == 'api':
            caps = self._capabilities(self.model)
            if caps is not None and 'vision' not in caps:
                return False, (f"model '{self.model}' cannot read images (capabilities: "
                               f"{', '.join(caps)}). Use a vision model for --vlm-model, "
                               f"e.g. qwen3-vl:8b or gemma4; a text-only model can "
                               f"only be used as --planner-model")
        planner = "" if self.planner == self.model else f", planner {self.planner}"
        return True, f"VLM ready: {self.model}{planner} via {self.backend}"

    def warm_up(self):
        """Load the model(s) into memory so the first real check is fast."""
        if self.backend != 'api':
            return
        for m in dict.fromkeys((self.model, self.planner)):
            try:
                self._post('/api/generate', {'model': m, 'keep_alive': VLM_KEEP_ALIVE},
                           self.timeout)
            except Exception as exc:          # not fatal - first call loads it
                print(f"VLM warm-up of {m} failed: {exc}")

    def ask(self, prompt, jpeg=None, schema=None, model=None):
        """Send prompt (+ optional image), return the raw text reply."""
        model = model or self.model
        if self.backend == 'cli':
            return self._ask_cli(model, prompt, jpeg)
        payload = {
            'model': model,
            'prompt': prompt,
            'format': schema if (schema and self.schemas_ok) else 'json',
            'stream': False,
            'keep_alive': VLM_KEEP_ALIVE,
            'options': VLM_OPTIONS,
        }
        if jpeg is not None:
            payload['images'] = [base64.b64encode(jpeg).decode()]
        if self.think_ok:
            # Qwen3-VL / Gemma 4 "think" first; the thinking eats the token
            # budget and the JSON answer comes back empty or cut off.
            payload['think'] = False
        try:
            reply = self._post('/api/generate', payload, self.timeout)
            text = reply.get('response', '')
            thinking = reply.get('thinking') or ''
            if not text.strip() and thinking:
                # Some Ollama versions put a thinking model's whole answer -
                # including the JSON - in 'thinking' and leave 'response' empty.
                if '{' in thinking:
                    return thinking[thinking.find('{'):]
                return ("[empty answer - the model only produced thinking: "
                        + thinking[:300] + "]")
            if reply.get('done_reason') == 'length':
                print("VLM: answer cut off at num_predict - raise VLM_OPTIONS['num_predict']")
            return text
        except urllib.error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode()
            except Exception:
                pass
            if exc.code == 400 and 'think' in body and self.think_ok:
                self.think_ok = False                    # model has no think switch
                return self.ask(prompt, jpeg, schema, model)
            if exc.code == 400 and isinstance(payload['format'], dict):
                print("Ollama rejected the JSON schema (update Ollama to >= 0.5); "
                      "falling back to plain JSON mode")
                self.schemas_ok = False
                return self.ask(prompt, jpeg, schema, model)
            raise

    def _ask_cli(self, model, prompt, jpeg):
        # `ollama run` attaches any image path that appears in the prompt.
        # The CLI only has plain JSON mode, so the schema is not enforced here.
        path = None
        try:
            if jpeg is not None:
                fd, path = tempfile.mkstemp(suffix='.jpg', prefix='vlm_')
                with os.fdopen(fd, 'wb') as f:
                    f.write(jpeg)
                prompt = f"{prompt}\n{path}"
            out = subprocess.run(
                ['ollama', 'run', model, '--format', 'json',
                 '--keepalive', VLM_KEEP_ALIVE, '--nowordwrap', prompt],
                capture_output=True, text=True, timeout=self.timeout)
            if out.returncode != 0:
                raise RuntimeError(out.stderr.strip() or "ollama run failed")
            return out.stdout
        finally:
            if path:
                os.remove(path)


def json_object(raw):
    """
    The JSON object in the reply as a dict, or {}. The reply may contain
    thinking text around it, so every '{' is tried and the LAST complete
    object is used (the final answer comes after any drafts).
    """
    dec, found, i = json.JSONDecoder(), {}, raw.find('{')
    while i != -1:
        try:
            data, end = dec.raw_decode(raw, i)
            if isinstance(data, dict):
                found = data
            i = raw.find('{', end)
        except json.JSONDecodeError:
            i = raw.find('{', i + 1)
    return found


def parse_look(raw, colours):
    """{'stacks': [[...]], 'table': [...]} or None if unusable."""
    data = json_object(raw)
    stacks_in = data.get('stacks')
    table_in = data.get('table')
    if not isinstance(stacks_in, list) or not isinstance(table_in, list):
        return None
    if stacks_in and all(isinstance(c, str) for c in stacks_in):
        stacks_in = [stacks_in]                       # one flat stack
    stacks, table = [], []
    for s in stacks_in:
        if not isinstance(s, list):
            continue
        cols = [to_colour(c, colours) for c in s]
        if None in cols:
            return None
        if len(cols) == 1:
            table += cols
        elif cols:
            stacks.append(cols)
    for c in table_in:
        col = to_colour(c, colours)
        if col is None:
            return None
        table.append(col)
    return {'stacks': stacks, 'table': table}


def parse_plan(raw, colours):
    """(status, problem, [(cube, to), ...]) or None if unusable."""
    data = json_object(raw)
    status = str(data.get('status', '')).strip().lower().replace(' ', '_')
    status = {'incomplete': 'in_progress', 'done': 'complete'}.get(status, status)
    moves_in = data.get('moves')
    if status not in PLAN_STATUSES or not isinstance(moves_in, list):
        return None
    moves = []
    for mv in moves_in:
        if not isinstance(mv, dict):
            return None
        cube = to_colour(mv.get('cube', ''), colours)
        to_raw = str(mv.get('to', '')).lower()
        to = 'table' if 'table' in to_raw else to_colour(to_raw, colours)
        if cube is None or to is None:
            return None
        moves.append((cube, to))
    problem = str(data.get('problem', '')).strip()
    if problem.lower().startswith('one sentence'):        # echoed the field description
        problem = ''
    return status, problem, moves


def code_plan(stacks, table):
    """
    Fallback planner: shortest-sensible moves to build the goal stack.
      1. keep the longest correct part of the goal already stacked (from the bottom)
      2. take everything above it down to the table
      3. place the remaining goal cubes in order, clearing anything on top first
    Returns [(cube, to), ...]; `to` is 'table' or a colour.
    """
    goal = goal_colours()
    piles = [list(s) for s in stacks] + [[c] for c in table]
    moves = []

    def move(cube, to):
        src = next(p for p in piles if p and p[-1] == cube)
        src.pop()
        if to == 'table':
            piles.append([cube])
        else:
            next(p for p in piles if p and p[-1] == to).append(cube)
        moves.append((cube, to))

    def clear_above(cube):
        pile = next(p for p in piles if cube in p)
        while pile[-1] != cube:
            move(pile[-1], 'table')

    def prefix(p):
        k = 0
        while k < min(len(p), len(goal)) and p[k] == goal[k]:
            k += 1
        return k

    base = max(piles, key=prefix, default=[])
    k = prefix(base)
    if k:
        while len(base) > k:                           # 2. unstack wrong cubes
            move(base[-1], 'table')
    for i in range(k, len(goal)):                      # 3. build upward
        cube = goal[i]
        if not any(cube in p for p in piles):
            break                                      # not visible - can't plan
        clear_above(cube)
        if i == 0:
            pile = next(p for p in piles if cube in p)
            if len(pile) > 1:                          # base must stand on the table
                move(cube, 'table')
        else:
            move(cube, goal[i - 1])
    return moves


def check_plan(stacks, table, status, moves):
    """
    Replay the VLM's moves on the accepted reading of the scene.
    Returns (instructions, error). instructions are the VLM's moves written
    out explicitly for the worker; error is None if the plan is valid.
    Nothing here chooses a move - it only checks the ones given.
    """
    goal = goal_colours()
    piles = [list(s) for s in stacks] + [[c] for c in table]

    done_now = any(p == goal for p in piles)
    if done_now and status != 'complete':
        return [], (f"a stack is already exactly {', '.join(goal)}, "
                    f"so the status must be complete with no moves")
    if status == 'complete' and not done_now:
        return [], (f"the status cannot be complete: no stack is exactly "
                    f"{', '.join(goal)} yet")
    if not done_now and not moves:
        return [], "the goal is not finished, so moves are needed"

    out = []
    for i, (cube, to) in enumerate(moves, 1):
        src = next((p for p in piles if p[-1] == cube), None)
        if src is None:
            where = next((p for p in piles if cube in p), None)
            if where is None:
                return out, f"move {i} is impossible: there is no {cube} cube"
            above = where[where.index(cube) + 1]
            return out, (f"move {i} is impossible: the {cube} cube has the {above} "
                         f"cube on top of it, which must be moved first")
        below = src[-2] if len(src) > 1 else None
        take = (f"Take the {cube.upper()} cube off the {below.upper()} cube"
                if below else f"Pick up the {cube.upper()} cube from the table")
        if to == 'table':
            if below is None:
                return out, f"move {i} does nothing: the {cube} cube is already on the table"
            src.pop()
            piles.append([cube])
            out.append(f"{take} and put it down on the table.")
        else:
            dst = next((p for p in piles if p is not src and p[-1] == to), None)
            if dst is None:
                return out, (f"move {i} is impossible: there is no {to} cube with a "
                             f"free top to put the {cube} cube on")
            src.pop()
            dst.append(cube)
            if not src:
                piles.remove(src)
            out.append(f"{take} and put it on top of the {to.upper()} cube.")

    if not any(p == goal for p in piles):
        final = "; ".join(", ".join(p) for p in piles if len(p) > 1) or "no stack"
        return out, (f"after all the moves the stacks are: {final}. The goal "
                     f"{', '.join(goal)} is not finished - more moves are needed")
    return out, None


def vlm_triggers(groups):
    """
    Reasons to call the VLM for this settle (empty list = no anomaly).
    The reading of YOLO + geometry is compared with the goal.
    """
    det = detector_state(groups)
    v = evaluate_state(det)
    why = list(v['anomalies'])
    if v['missing']:
        why.append(f"goal cube(s) not detected anywhere: {', '.join(v['missing'])}")
    for g in groups:
        if g['is_stack'] and not g['gaps_ok']:
            why.append(f"gap in the stack {'>'.join(normalize_label(c) for c in g['order'])}"
                       f" - a cube may be hidden")
    return why


def plan_for(state, verdict):
    """(status, problem, steps, moves, plan_source) worked out in code."""
    status = verdict['status']
    if status == 'complete':
        return status, problem_text(verdict, state), [], [], 'none needed'
    if verdict['missing']:
        return (status, (f"The {', '.join(verdict['missing'])} cube(s) cannot be seen. "
                         + (problem_text(verdict, state) if verdict['anomalies'] else "")
                         ).strip(),
                [f"Put the {c.upper()} cube where the camera can see it."
                 for c in verdict['missing']], [], 'code (cube not visible)')
    moves = code_plan(state['stacks'], state['table'])
    steps, _ = check_plan(state['stacks'], state['table'], status, moves)
    return (status, problem_text(verdict, state), steps,
            [{'cube': c, 'to': t} for c, t in moves], 'code')


def code_result(groups, why_no_vlm):
    """Same shape as run_vlm_check()'s result, from YOLO + geometry only."""
    det = detector_state(groups)
    verdict = evaluate_state(det)
    status, problem, steps, moves, source = plan_for(det, verdict)
    return {'detections': short_summary(groups), 'model': None, 'planner': None,
            'vlm_called': False, 'why_no_vlm': why_no_vlm, 'error': None,
            'detector_state': det, 'detector_verdict': verdict,
            'vlm_state': None, 'vlm_verdict': None, 'look_problems': [],
            'look_attempts': 0, 'comparison': None, 'vlm_boxes': {},
            'state': det, 'state_source': 'YOLO + geometry', 'verdict': verdict,
            'parsed': True, 'status': status, 'problem': problem, 'steps': steps,
            'moves': moves, 'verified': True, 'plan_source': source,
            'planner_status': None, 'plan_error': None, 'attempts': 0,
            'raw': [], 'seconds': 0.0}


def vlm_look(client, frame, groups, crop=None, result=None):
    """
    LOOK step: the VLM reads stacks / table cubes bottom -> top from the image.
    Re-asked (VLM_LOOK_RETRIES) with the reason if the reading contradicts
    sure facts. Returns (reading or None, problems). Fills `result` if given.
    """
    result = result if result is not None else {'raw': []}
    colours = known_colours(groups)
    jpeg = prepare_vlm_image(frame, groups, crop)
    result['jpeg'] = jpeg
    base = look_prompt(groups, colours)
    prompt, seen, probs = base, None, []
    for attempt in range(1, VLM_LOOK_RETRIES + 2):
        result['look_attempts'] = attempt
        raw = client.ask(prompt, jpeg, look_schema(colours))
        result['raw'].append(raw)
        seen = parse_look(raw, colours)
        probs = (["the answer was not a valid list of stacks and table cubes"]
                 if seen is None else look_problems(seen, groups))
        if not probs:
            break
        prompt = (f"{base}\n\nYour previous reading was wrong:\n"
                  + "\n".join(f"- {p}" for p in probs)
                  + "\nLook at the photo again and give the full, corrected reading.")
    result['look_problems'] = probs
    return seen, probs


def prepare_square_image(frame, crop=None, side=None):
    """
    Frame (or work-area crop) scaled to fit and padded to a side x side square.
    In a square image pixel coordinates and Qwen's 0-1000 relative coordinates
    are the same, so either convention the model uses maps back correctly.
    Returns (jpeg, ox, oy, scale).
    """
    side = side or VLM_GROUND_SIDE
    ox, oy, img = 0, 0, frame
    if crop is not None and VLM_CROP_TO_WORK_AREA:
        x1, y1, x2, y2 = crop
        if x2 - x1 > 32 and y2 - y1 > 32:
            img, ox, oy = frame[y1:y2, x1:x2], x1, y1
    h, w = img.shape[:2]
    scale = side / max(h, w)
    img = cv2.resize(img, (round(w * scale), round(h * scale)),
                     interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    canvas = np.full((side, side, 3), 114, np.uint8)
    canvas[:img.shape[0], :img.shape[1]] = img
    ok, buf = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("could not encode frame for the VLM")
    return buf.tobytes(), ox, oy, scale


def vlm_ground(client, frame, need, state, crop=None, result=None):
    """
    GROUND step: ask the VLM for the box of each cube it found but YOLO did
    not. Returns {colour: (x1, y1, x2, y2)} in frame pixels. The boxes are
    only suggestions - final_boxes() checks them against colour and size.
    """
    colours = list(dict.fromkeys(need))
    side = VLM_GROUND_SIDE
    jpeg, ox, oy, scale = prepare_square_image(frame, crop, side)
    context = "; ".join([f"a stack, from bottom to top: {', '.join(s)}"
                         for s in state['stacks']]
                        + [f"a single {c} cube on the table" for c in state['table']])
    prompt = f"""The photo shows coloured cubes on a table: {context}.
Find these cubes: {', '.join(colours)}.
For each one give "box": [x1, y1, x2, y2] - the tight box around the whole visible cube
(its front side and its top), in pixel coordinates of this {side}x{side} image:
x grows to the right, y grows downward, (0, 0) is the top-left corner."""
    col = {'type': 'string', 'enum': colours}
    schema = {'type': 'object', 'properties': {'boxes': {'type': 'array', 'items': {
        'type': 'object', 'properties': {
            'colour': col,
            'box': {'type': 'array', 'items': {'type': 'number'},
                    'minItems': 4, 'maxItems': 4}},
        'required': ['colour', 'box']}}}, 'required': ['boxes']}
    raw = client.ask(prompt, jpeg, schema)
    if result is not None:
        result['raw'].append(raw)
    out = {}
    h_img, w_img = frame.shape[:2]
    for item in json_object(raw).get('boxes', []) or []:
        if not isinstance(item, dict):
            continue
        colour = to_colour(item.get('colour', ''), colours)
        box = item.get('box') or item.get('bbox_2d')
        if colour is None or colour in out or not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(v) for v in box)
        except (TypeError, ValueError):
            continue
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        fb = (int(max(0, ox + x1 / scale)), int(max(0, oy + y1 / scale)),
              int(min(w_img - 1, ox + x2 / scale)), int(min(h_img - 1, oy + y2 / scale)))
        if fb[2] - fb[0] > 4 and fb[3] - fb[1] > 4:
            out[colour] = fb
    return out


def run_vlm_check(client, frame, groups, crop=None):
    """LOOK (complements YOLO) + order check + PLAN with verification."""
    colours = known_colours(groups)
    det = detector_state(groups)
    result = {'detections': short_summary(groups), 'model': client.model,
              'planner': client.planner, 'error': None,
              'detector_state': det, 'detector_verdict': evaluate_state(det),
              'vlm_state': None, 'vlm_verdict': None, 'look_problems': [],
              'look_attempts': 0, 'comparison': None,
              'state': det, 'state_source': 'detector', 'verdict': None,
              'parsed': False, 'status': None, 'problem': '', 'steps': [],
              'moves': [], 'verified': False, 'plan_source': None,
              'planner_status': None, 'plan_error': None, 'attempts': 0,
              'raw': []}
    result['vlm_boxes'] = {}
    result['vlm_called'] = True
    result['triggers'] = vlm_triggers(groups)
    t0 = time.time()
    try:
        # ---- 1. LOOK: the vision model reads the image ----
        seen, probs = vlm_look(client, frame, groups, crop, result)
        if seen is not None:
            result['vlm_state'] = seen
            result['vlm_verdict'] = evaluate_state(seen)
            result['comparison'] = compare_readings(det, seen)
        if seen is not None and not probs:
            result['state'] = seen
            result['state_source'] = ('YOLO + VLM agree' if same_state(det, seen)
                                      else 'VLM corrected YOLO')
            # 1b. GROUND: boxes for the cubes only the VLM found
            need = list((Counter(all_colours(seen)) - Counter(all_colours(det))).elements())
            if need and VLM_GROUND:
                result['vlm_boxes'] = vlm_ground(client, frame, need, seen, crop, result)
        else:
            result['state_source'] = 'YOLO only (VLM reading rejected)'

        # ---- 2. order check on the accepted reading ----
        state = result['state']
        verdict = evaluate_state(state)
        result['verdict'] = verdict
        expected = verdict['status']

        if expected == 'complete':
            result.update(parsed=True, status='complete', verified=True,
                          plan_source='none needed')
        elif verdict['missing']:
            # A goal cube is not visible anywhere - no plan can finish the goal.
            result.update(
                parsed=True, status=expected, verified=True,
                plan_source='code (cube not visible)',
                problem=(f"The {', '.join(verdict['missing'])} cube(s) cannot be seen. "
                         + (problem_text(verdict, state) if verdict['anomalies'] else "")
                         ).strip(),
                steps=[f"Put the {c.upper()} cube where the camera can see it."
                       for c in verdict['missing']])
        else:
            # ---- 3. PLAN: text only, verified, retried with the reason ----
            why = ('; '.join(verdict['anomalies']) if expected == 'wrong' else
                   'no stack has a cube in the wrong place - cubes are only missing')
            base = plan_prompt(state['stacks'], state['table'], colours)
            prompt, best, last = base, None, None
            for attempt in range(1, VLM_PLAN_RETRIES + 2):
                result['attempts'] = attempt
                raw = client.ask(prompt, None, plan_schema(colours), client.planner)
                result['raw'].append(raw)
                plan = parse_plan(raw, colours)
                if plan is None:
                    err = "the answer did not have a valid status and list of moves"
                else:
                    status, problem, moves = plan
                    result['planner_status'] = status
                    steps, err = check_plan(state['stacks'], state['table'],
                                            status, moves)
                    last = (status, problem, moves, steps)
                    if err is None:
                        best = last
                        if status == expected:
                            break
                        err = (f"the moves are fine, but the status must be "
                               f"{expected}, not {status}: {why}")
                result['plan_error'] = err
                prompt = (f"{base}\n\nYour previous answer was wrong: {err}.\n"
                          f"Think again and give the full, correct answer.")

            if best is not None:
                status, problem, moves, steps = best
                plan_source = 'VLM'
                result['plan_error'] = None
            else:
                # The VLM gave no valid plan: compute the moves in code, so the
                # worker always gets step-by-step instructions.
                status, problem = expected, ''
                moves = code_plan(state['stacks'], state['table'])
                steps, err = check_plan(state['stacks'], state['table'], expected, moves)
                plan_source = 'code (VLM plan unusable)'
                if err is not None:                      # should never happen
                    result['plan_error'] = f"code plan failed: {err}"
            result.update(
                parsed=True, steps=steps, verified=True, plan_source=plan_source,
                moves=[{'cube': c, 'to': t} for c, t in moves],
                problem=(problem if status == expected and problem
                         else problem_text(verdict, state)))
            result['status'] = expected      # the order check decides the header
    except Exception as exc:
        result['error'] = f"{type(exc).__name__}: {exc}"
    if result['verdict'] is None:
        result['verdict'] = evaluate_state(result['state'])
    result['seconds'] = round(time.time() - t0, 1)
    return result


def format_vlm_report(res):
    """Multi-line terminal report."""
    bar = "=" * 70
    if not res.get('vlm_called', True):
        v = res['verdict']
        return "\n".join([bar, f"NO VLM ({res['why_no_vlm']})",
                          f"  YOLO+geom: {state_summary(res['state'])}  -> {v['status']}"]
                         + [f"    {i}. {st}" for i, st in enumerate(res['steps'], 1)]
                         + [bar])
    who = res['model'] + ("" if res['planner'] == res['model'] else f" + {res['planner']}")
    lines = [bar, f"VLM ({who}, {res['seconds']} s, look x{res['look_attempts']}, "
                  f"plan x{res['attempts']})",
             f"  YOLO     : {state_summary(res['detector_state'])}"]
    if res['vlm_state'] is not None:
        lines.append(f"  VLM      : {state_summary(res['vlm_state'])}")
    if res['look_problems']:
        lines.append(f"  rejected : {'; '.join(res['look_problems'])}")
    lines.append(f"  final    : {state_summary(res['state'])}  [{res['state_source']}]")
    cmp = res.get('comparison')
    if cmp and not cmp['same_reading']:
        lines.append(f"  compare  : missed by YOLO {cmp['missed_by_detector'] or '-'}, "
                     f"missed by VLM {cmp['missed_by_vlm'] or '-'}, "
                     f"placed differently {[m['cube'] for m in cmp['placed_differently']] or '-'}")
    if res['error']:
        return "\n".join(lines + [f"  ERROR    : {res['error']}", bar])
    v = res['verdict']
    lines += [f"  order    : {v['status']}"
              + (f" - {'; '.join(v['anomalies'])}" if v['anomalies'] else "")
              + (f" (not visible: {', '.join(v['missing'])})" if v['missing'] else "")]
    if not res['parsed']:
        return "\n".join(lines + ["  no usable plan - raw replies:"]
                         + [f"    {r.strip()[:300]}" for r in res['raw']] + [bar])
    lines += [f"  plan     : {res['plan_source']}"
              + (f" (planner said {res['planner_status']})" if res['planner_status'] else ""),
              f"  problem  : {res['problem'] or '-'}",
              "  steps    :"]
    lines += [f"    {i}. {s}" for i, s in enumerate(res['steps'], 1)] or ["    -"]
    if res['plan_error']:
        lines.append(f"  check    : {res['plan_error']}")
    return "\n".join(lines + [bar])


# --------------------------------------------------------------------------
# Per-settle records (image + JSON, bottom -> top)
# --------------------------------------------------------------------------


def detector_json(groups):
    """YOLO boxes per stack, bottom -> top, with confidence and source."""
    tags = group_tags(groups)

    def cube(c):
        return {'colour': normalize_label(c['label']),
                'confidence': round(c['conf'], 3),
                'source': 'colour_probe' if c.get('inferred') else 'yolo',
                'yolo_label': c.get('yolo_label'),
                'recoloured': bool(c.get('recoloured')),
                'side_face': is_side_face(c),
                'merged_duplicates': c.get('merged', []),
                'box_xyxy': [int(c['x1']), int(c['y1']), int(c['x2']), int(c['y2'])]}

    return {'stacks': [{'id': tags[id(g)],
                        'bottom_to_top': [cube(c) for c in g['members']],
                        'gap_warning': not g['gaps_ok'],
                        'spread_warning': not g['x_spread_ok']}
                       for g in groups if g['is_stack']],
            'table': [cube(g['members'][0]) for g in groups if not g['is_stack']]}


def yolo_only_json(cubes):
    """Plain YOLO boxes, ungrouped, sorted bottom (lowest in image) -> top."""
    return [{'colour': normalize_label(c['label']), 'confidence': round(c['conf'], 3),
             'box_xyxy': [int(c['x1']), int(c['y1']), int(c['x2']), int(c['y2'])]}
            for c in sorted(cubes, key=lambda c: -c['cy'])]


def build_record(settle_id, images, groups, res=None, note=None, yolo_only=()):
    """Everything about one settled scene, ready for json.dump."""
    det = detector_state(groups)
    det_verdict = evaluate_state(det)
    rec = {'timestamp': datetime.now().isoformat(timespec='seconds'),
           'settle_id': settle_id,
           'images': images,
           'goal_bottom_to_top': goal_colours(),
           'goal_source': GOAL_SOURCE,
           'yolo_only': yolo_only_json(yolo_only),
           'detector': {'description': 'YOLO + colour check + geometric grouping '
                                       '+ hidden-cube inference',
                        'order': det, 'verdict': det_verdict, **detector_json(groups)},
           'vlm': None,
           'comparison': None}
    if note:
        rec['note'] = note
    if res is None:
        rec['final'] = {'source': 'YOLO only', 'order': det, 'verdict': det_verdict,
                        'status': det_verdict['status'],
                        'problem': problem_text(det_verdict, det),
                        'instructions': [], 'plan_verified': False}
        return rec
    if not res.get('vlm_called', True):
        rec['vlm'] = {'called': False, 'why_not': res['why_no_vlm']}
    else:
        rec['vlm'] = {'called': True, 'triggered_by': res.get('triggers', []),
                  'model': res['model'], 'planner': res['planner'],
                  'order': res['vlm_state'], 'verdict': res['vlm_verdict'],
                  'reading_accepted': res['vlm_state'] is not None
                                      and not res['look_problems'],
                  'rejected_because': res['look_problems'],
                  'look_attempts': res['look_attempts'],
                  'suggested_boxes': {k: list(v) for k, v in
                                      (res.get('vlm_boxes') or {}).items()},
                  'raw_replies': res['raw']}
    rec['comparison'] = res['comparison']
    rec['final'] = {'source': res['state_source'], 'order': res['state'],
                    'verdict': res['verdict'],
                    'status': res['status'] or res['verdict']['status'],
                    'problem': res['problem'] or problem_text(res['verdict'], res['state']),
                    'instructions': res['steps'], 'moves': res['moves'],
                    'plan_source': res['plan_source'],
                    'plan_verified': res['verified'],
                    'planner_status': res['planner_status'],
                    'plan_attempts': res['attempts'],
                    'plan_error': res['plan_error']}
    rec['error'] = res['error']
    rec['seconds'] = res.get('seconds')
    return rec


IMG_RAW, IMG_YOLO, IMG_GEOM, IMG_FINAL = ('1_raw.jpg', '2_yolo_only.jpg',
                                          '3_yolo_geometric.jpg', '4_final_vlm.jpg')


def draw_yolo_only(image, cubes):
    """Plain YOLO output: one box per detection with class and confidence."""
    for c in cubes:
        colour = cube_color(c['label'])
        edge = contrast_color(colour)
        box = (c['x1'], c['y1'], c['x2'], c['y2'])
        cv2.rectangle(image, box[:2], box[2:], edge, 4)
        cv2.rectangle(image, box[:2], box[2:], colour, 2)
        draw_label(image, f"{normalize_label(c['label'])} {c['conf']:.2f}", box,
                   colour, edge, 0.45, 1)
    return image


def stage_title(image, text):
    """Stage name in the bottom-left corner of a saved image."""
    y = image.shape[0] - 12
    (tw, th), base = cv2.getTextSize(text, FONT, 0.6, 1)
    blend_rect(image, 0, y - th - 10, tw + 20, image.shape[0], PANEL_BG, 0.85)
    cv2.putText(image, text, (10, y), FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image


def save_jpg(path, image):
    if not cv2.imwrite(str(path), image):
        print(f"could not save {path}")


def save_stage_images(folder, frame, yolo_only, groups):
    """Images 1-3 of a settle. Returns the file-name map for the JSON."""
    folder.mkdir(parents=True, exist_ok=True)
    save_jpg(folder / IMG_RAW, frame)
    save_jpg(folder / IMG_YOLO, stage_title(
        draw_yolo_only(frame.copy(), yolo_only),
        f"YOLO only: {len(yolo_only)} box(es)"))
    save_jpg(folder / IMG_GEOM, stage_title(
        draw_detections(frame.copy(), groups),
        f"YOLO + geometry: {short_summary(groups)}"))
    return {'raw': IMG_RAW, 'yolo_only': IMG_YOLO, 'yolo_geometric': IMG_GEOM,
            'final_vlm': IMG_FINAL}


VLM_OFF_REASON = None        # set in main() when the VLM could not be started



def vlm_reading_used(res):
    """True when the final reading came from the VLM (not the YOLO fallback)."""
    return (res is not None and res.get('vlm_state') is not None
            and not res.get('look_problems'))


def colour_fraction(frame, colour, box):
    """Fraction of the central half of `box` whose pixels match `colour`."""
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) // 4, (y2 - y1) // 4
    roi = frame[max(0, y1 + my):max(0, y2 - my), max(0, x1 + mx):max(0, x2 - mx)]
    if roi.size == 0:
        return 0.0
    masks = color_masks(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(int))
    return float(masks[colour].mean()) if colour in masks else 0.0


def colour_fit(frame, colour, guess, ref_w, ref_h):
    """
    Box of the `colour` blob near `guess`, sized like a cube, or None.
    Searches a window around the guess, keeps the largest connected region of
    that colour whose size fits a cube (so a white wall or a dark shadow that
    spreads far beyond a cube is not taken).
    """
    if colour in NO_COLOUR_CHECK:
        return None
    h_img, w_img = frame.shape[:2]
    gx1, gy1, gx2, gy2 = guess
    wx1, wy1 = int(max(0, gx1 - 0.5 * ref_w)), int(max(0, gy1 - 0.6 * ref_h))
    wx2, wy2 = int(min(w_img, gx2 + 0.5 * ref_w)), int(min(h_img, gy2 + 0.6 * ref_h))
    roi = frame[wy1:wy2, wx1:wx2]
    if roi.size == 0:
        return None
    masks = color_masks(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).astype(int))
    if colour not in masks:
        return None
    m = masks[colour].astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(m)
    best, best_area = None, 0
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        if not (0.5 * ref_w <= w <= 1.6 * ref_w and 0.35 * ref_h <= h <= 1.6 * ref_h):
            continue
        if area < 0.25 * ref_w * ref_h or area <= best_area:
            continue
        best, best_area = (wx1 + x, wy1 + y, wx1 + x + w, wy1 + y + h), area
    return best


def vlm_box_ok(frame, colour, box, ref_w, ref_h, guess=None):
    """A VLM-suggested box is used only if size, place and colour all fit."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if not (0.5 * ref_w <= w <= 1.8 * ref_w and 0.4 * ref_h <= h <= 1.8 * ref_h):
        return False
    if guess is not None:
        gcx, gcy = (guess[0] + guess[2]) / 2, (guess[1] + guess[3]) / 2
        if abs((x1 + x2) / 2 - gcx) > ref_w or abs((y1 + y2) / 2 - gcy) > 1.2 * ref_h:
            return False
    if colour in NO_COLOUR_CHECK:      # e.g. white looks like the wall
        return True
    return colour_fraction(frame, colour, box) >= 0.25


def final_boxes(groups, state, frame, vlm_boxes=None):
    """
    Boxes for the FINAL reading (bottom -> top per stack).
      - every colour gets the detected box of that colour (YOLO or colour probe)
        that fits the stack best: closest to the column of the cubes already placed
      - a cube found only by the VLM gets, in this order:
          1. the VLM's own box, if its size, place and colour check out  -> 'vlm_box'
          2. the cube-sized blob of its colour near where it must be     -> 'vlm_colour_fit'
          3. one cube-step above / below its neighbour (estimate)        -> 'vlm_estimate'
      - a single table cube found only by the VLM: its VLM box if it checks
        out, otherwise listed as unplaced
    Returns (entries, ignored_detections, unplaced_colours).
    """
    vlm_boxes = vlm_boxes or {}
    h_img, w_img = frame.shape[:2]
    pool = [c for g in groups for c in g['members']]
    full = [c for c in pool if not is_side_face(c)] or pool
    used = set()
    entries, unplaced = [], []

    def take(colour, near_x=None):
        cands = [c for c in pool if id(c) not in used
                 and normalize_label(c['label']) == colour]
        if not cands:
            return None
        if near_x is None:      # prefer full faces, then confidence
            best = max(cands, key=lambda c: (not is_side_face(c), c['conf']))
        else:
            best = min(cands, key=lambda c: abs(c['cx'] - near_x))
        used.add(id(best))
        return best

    def ref_size(neighbour=None):
        if full:
            return (float(np.median([c['w'] for c in full])),
                    float(np.median([c['h'] for c in full])))
        if neighbour is not None:
            return neighbour[2] - neighbour[0], neighbour[3] - neighbour[1]
        return w_img / 10, h_img / 6

    def place_vlm(colour, guess, ref_w, ref_h):
        vb = vlm_boxes.get(colour)
        if vb is not None and vlm_box_ok(frame, colour, vb, ref_w, ref_h, guess):
            return (*vb, 'vlm_box')
        if guess is None:
            return None
        fit = colour_fit(frame, colour, guess, ref_w, ref_h)
        if fit is not None:
            return (*fit, 'vlm_colour_fit')
        return (*guess, 'vlm_estimate')

    for si, stack in enumerate(state['stacks'], 1):
        slots = [None] * len(stack)
        # place colours that have a single candidate first: they anchor the column
        order = sorted(range(len(stack)), key=lambda i: sum(
            1 for c in pool if normalize_label(c['label']) == stack[i]))
        for i in order:
            placed = [s for s in slots if s is not None]
            near = float(np.mean([s['cx'] for s in placed])) if placed else None
            slots[i] = take(stack[i], near)
        boxes = [None if s is None else
                 (s['x1'], s['y1'], s['x2'], s['y2'], 'yolo' if not s.get('inferred')
                  else 'colour_probe') for s in slots]
        ref_w, ref_h = ref_size(next((b for b in boxes if b), None))

        # vertical step between cubes: from placed members if two are known
        known = [(i, (b[1] + b[3]) / 2) for i, b in enumerate(boxes) if b]
        steps = [(known[k][1] - known[k + 1][1]) / (known[k + 1][0] - known[k][0])
                 for k in range(len(known) - 1)]
        step = float(np.median(steps)) if steps else 0.85 * ref_h

        # no detected cube at all in this stack: start from any VLM box
        if not known:
            for i, colour in enumerate(stack):
                vb = vlm_boxes.get(colour)
                if vb is not None and vlm_box_ok(frame, colour, vb, ref_w, ref_h):
                    boxes[i] = (*vb, 'vlm_box')
                    break

        changed = True
        while changed and any(b is None for b in boxes):
            changed = False
            for i, b in enumerate(boxes):
                if b is not None:
                    continue
                for j, sign in ((i - 1, -1), (i + 1, 1)):   # below -> step up
                    if 0 <= j < len(boxes) and boxes[j] is not None:
                        x1, y1, x2, y2, _ = boxes[j]
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + sign * step
                        guess = (int(cx - ref_w / 2), int(max(0, cy - ref_h / 2)),
                                 int(cx + ref_w / 2), int(min(h_img - 1, cy + ref_h / 2)))
                        if guess[3] - guess[1] > 5:
                            boxes[i] = place_vlm(stack[i], guess, ref_w, ref_h)
                            changed = True
                        break
        for i, colour in enumerate(stack):
            if boxes[i] is None:
                unplaced.append(colour)
                continue
            x1, y1, x2, y2, src = boxes[i]
            entries.append({'colour': colour, 'stack': f"S{si}", 'position': i + 1,
                            'source': src, 'box_xyxy': [int(x1), int(y1), int(x2), int(y2)]})
    for colour in state['table']:
        c = take(colour)
        if c is not None:
            entries.append({'colour': colour, 'stack': None, 'position': None,
                            'source': 'colour_probe' if c.get('inferred') else 'yolo',
                            'box_xyxy': [int(c['x1']), int(c['y1']),
                                         int(c['x2']), int(c['y2'])]})
            continue
        ref_w, ref_h = ref_size()
        b = place_vlm(colour, None, ref_w, ref_h)
        if b is None:
            unplaced.append(colour)
        else:
            entries.append({'colour': colour, 'stack': None, 'position': None,
                            'source': b[4], 'box_xyxy': [int(v) for v in b[:4]]})
    ignored = [c for c in pool if id(c) not in used]
    return entries, ignored, unplaced


def draw_entries(image, entries):
    """Boxes from final_boxes(): YOLO solid, everything else dashed + tagged."""
    for e in entries:
        x1, y1, x2, y2 = e['box_xyxy']
        colour = cube_color(e['colour'])
        edge = contrast_color(colour)
        tag = f"{e['stack']}-{e['position']} {e['colour']}" if e['stack'] else e['colour']
        if e['source'] == 'yolo':
            cv2.rectangle(image, (x1, y1), (x2, y2), edge, 4)
            cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        else:
            dashed_rect(image, (x1, y1), (x2, y2), edge, 4)
            dashed_rect(image, (x1, y1), (x2, y2), colour, 2)
            tag += (" (VLM)" if e['source'].startswith('vlm')
                    else " (geometric reasoning)")
        draw_label(image, tag, (x1, y1, x2, y2), colour, edge, 0.45, 1)
    return image


def draw_final(image, groups, res):
    """
    Draw the final reading. With an accepted VLM reading: its cubes, VLM-only
    ones dashed and tagged '(VLM)'; detections it rejected are not drawn.
    Without one: the YOLO + geometry detections as usual.
    `image` must still be clean (boxes are fitted to its pixels).
    """
    if not vlm_reading_used(res):
        return draw_detections(image, groups)
    entries, _, _ = final_boxes(groups, res['state'], image, res.get('vlm_boxes'))
    return draw_entries(image, entries)


def render_final(frame, groups, res, footer="", note=None):
    """Image 4: the final reading's boxes and the VLM panel (as on the GUI)."""
    img = draw_final(frame.copy(), groups, res)
    if res is not None:
        header, colour, problem, steps, notes = instructions_from_result(res)
        title = (f"YOLO + geometry + VLM ({res['model']})" if res.get('vlm_called', True)
                 else "YOLO + geometry (no anomaly - VLM not called)")
    else:
        det = detector_state(groups)
        why = note or VLM_OFF_REASON or "VLM off"
        header, colour, problem, steps, notes = (
            "NO VLM RESULT", COLOR_BAD,
            problem_text(evaluate_state(det), det), [],
            [f"Seen (bottom>top): {state_summary(det)}", f"Why no VLM: {why}"])
        title = "YOLO + geometry (no VLM)"
    draw_instruction_panel(img, header, colour, problem, steps, footer or title,
                           COLOR_IDLE, side=choose_panel_side(groups, img.shape[1]),
                           notes=notes)
    return stage_title(img, title)


def finish_settle(folder, settle_id, frame, groups, yolo_only, images,
                  res=None, note=None, log_path=None):
    """Image 4 + result.json (+ one log line). Returns the JSON path."""
    if res is None and note is None:
        note = f"VLM not run: {VLM_OFF_REASON or 'disabled (--no-vlm)'}"
    save_jpg(folder / IMG_FINAL, render_final(frame, groups, res, note=note))
    rec = build_record(settle_id, images, groups, res, note, yolo_only)
    if vlm_reading_used(res):
        entries, ignored, unplaced = final_boxes(groups, res['state'], frame,
                                                 res.get('vlm_boxes'))
        rec['final']['boxes_bottom_to_top'] = entries
        rec['final']['ignored_detections'] = [
            {'colour': normalize_label(c['label']), 'confidence': round(c['conf'], 3),
             'box_xyxy': [int(c['x1']), int(c['y1']), int(c['x2']), int(c['y2'])]}
            for c in ignored]
        rec['final']['unplaced'] = unplaced
    return write_record(folder, rec, log_path)


def write_record(folder, record, log_path=None):
    """result.json in the settle folder, plus one log line."""
    path = folder / "result.json"
    path.write_text(json.dumps(record, indent=2))
    if log_path is not None:
        f = record['final']
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        text = (f"ERROR {record.get('error')}" if record.get('error') else
                f"{f['status']} [{f['source']}] {state_summary(f['order'])}: "
                f"{f['problem']} | "
                + " ".join(f"{i}) {s}" for i, s in enumerate(f['instructions'], 1)))
        with open(log_path, 'a') as fh:
            fh.write(f"{ts}  [settle {record['settle_id']}] {text}\n")
    return path


class VLMWorker:
    """
    Runs VLM checks in a background thread so the camera loop never waits.
    EVERY settle is checked, in order (nothing is dropped, even if the hand
    moves again or the model is still loading). Each result carries the settle
    id it answers, so an answer about an old scene is never shown as current.
    """

    def __init__(self, client, output_folder, log_path):
        self.client = client
        self.output_folder, self.log_path = output_folder, log_path
        self.jobs = queue.Queue()
        self.lock = threading.Lock()
        self.result = None               # most recent finished result
        self.busy = False
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, job):
        """job: frame, groups, yolo_only, images, folder, settle_id, crop."""
        self.jobs.put(job)
        if self.jobs.qsize() > 1:
            print(f"VLM: {self.jobs.qsize()} settles waiting")

    def _finish(self, job, res=None, note=None):
        try:
            path = finish_settle(job['folder'], job['settle_id'], job['frame'],
                                 job['groups'], job['yolo_only'], job['images'],
                                 res, note, self.log_path)
            print(f"Saved {path.parent.name}/ (4 images + result.json)")
        except Exception:
            print(f"could not save settle {job['settle_id']}:\n{traceback.format_exc()}")

    def latest(self, settle_id=None):
        """Result for this settle (or the most recent one if settle_id is None)."""
        with self.lock:
            if self.result is not None and (settle_id is None
                                            or self.result['settle_id'] == settle_id):
                return self.result
        return None

    def thinking(self):
        with self.lock:
            return self.busy or not self.jobs.empty()

    def _loop(self):
        self.client.warm_up()
        while True:
            job = self.jobs.get()
            with self.lock:
                self.busy = True
            res = None
            try:
                res = run_vlm_check(self.client, job['frame'], job['groups'],
                                    job['crop'])
                res['settle_id'] = job['settle_id']
                print(format_vlm_report(res))
                self._finish(job, res)
            except Exception:
                # Never let one bad settle kill the thread (that left later
                # settles with only images 1-3 and no JSON).
                print(f"VLM: settle {job['settle_id']} failed:\n{traceback.format_exc()}")
                self._finish(job, note="VLM check crashed - see terminal")
            finally:
                with self.lock:
                    self.result = res if res is not None else self.result
                    self.busy = False
                self.jobs.task_done()

    def drain(self, timeout=VLM_TIMEOUT):
        """On quit: finish checks still queued, so every settle gets its JSON."""
        left = self.jobs.qsize() + (1 if self.busy else 0)
        if not left:
            return
        print(f"Finishing {left} VLM check(s) before exit (Ctrl+C to skip) ...")
        end = time.time() + timeout
        try:
            while (self.busy or not self.jobs.empty()) and time.time() < end:
                time.sleep(0.2)
        except KeyboardInterrupt:
            pass
        while True:                          # anything left: save without VLM
            try:
                job = self.jobs.get_nowait()
            except queue.Empty:
                break
            self._finish(job, note="VLM skipped: program closed before the check ran")


# --------------------------------------------------------------------------
# Motion gating
# --------------------------------------------------------------------------

IDLE, MOVING, SETTLING, CONFIRMING = 'IDLE', 'MOVING', 'SETTLING', 'CONFIRMING'


class MotionDetector:
    """
    Measures how much of the work area differs from an ANCHOR frame.

    Comparing with an anchor rather than only the previous frame matters: a
    slow or briefly paused hand changes little between consecutive frames and
    would look "still". Held against the frame where settling began, it
    drifts past the threshold and restarts the settle clock.
    Colour (max over B, G, R) is compared, not grey, so a hand whose
    brightness happens to match the background is still seen.
    """

    def __init__(self, work_area=WORK_AREA):
        self.work_area = work_area       # frame fractions (x1, y1, x2, y2) or None
        self.anchor = None
        self.level = 0.0                 # last changed-pixel fraction, for display

    def reset(self):
        self.anchor, self.level = None, 0.0

    def area_px(self, frame):
        """Work area in pixels (x1, y1, x2, y2)."""
        h, w = frame.shape[:2]
        if self.work_area is None:
            return 0, 0, w, h
        fx1, fy1, fx2, fy2 = self.work_area
        return int(fx1 * w), int(fy1 * h), int(fx2 * w), int(fy2 * h)

    def prepare(self, frame):
        x1, y1, x2, y2 = self.area_px(frame)
        small = cv2.resize(frame[y1:y2, x1:x2], None, fx=MOTION_SCALE,
                           fy=MOTION_SCALE, interpolation=cv2.INTER_AREA)
        return cv2.GaussianBlur(small, (5, 5), 0).astype(np.int16)

    def measure(self, frame):
        """Update self.level against the anchor; returns the prepared frame."""
        cur = self.prepare(frame)
        if self.anchor is None or self.anchor.shape != cur.shape:
            self.anchor, self.level = cur, 0.0
            return cur
        changed = np.abs(cur - self.anchor).max(axis=2) > MOTION_PIXEL_THRESH
        self.level = float(changed.mean())
        return cur

    @property
    def changed(self):
        return self.level > MOTION_AREA_FRAC

    def set_anchor(self, prepared):
        self.anchor = prepared


class Gate:
    """Decides, frame by frame, whether YOLO should run."""

    def __init__(self):
        self.state = CONFIRMING          # evaluate the scene once at start-up
        self.after_motion = True         # this confirm window follows real motion
        self.still_since = None
        self.last_check = 0.0
        self.confirm_frames = 0
        self.hot = 0                     # IDLE: consecutive changed frames

    def start_confirm(self, after_motion=True):
        self.state, self.confirm_frames = CONFIRMING, 0
        self.after_motion = after_motion

    def done(self, now):
        self.state, self.last_check = IDLE, now

    def step(self, motion, frame, now):
        """Advance the state machine; True means run YOLO on this frame."""
        cur = motion.measure(frame)
        changed = motion.changed

        if self.state == MOVING:
            motion.set_anchor(cur)       # follow the motion frame to frame
            if not changed:              # looks still: anchor is now fixed here
                self.state, self.still_since = SETTLING, now
            return False

        if self.state == SETTLING:
            if changed:                  # drifted from the anchor: restart clock
                motion.set_anchor(cur)
                self.still_since = now
                return False
            if now - self.still_since < SETTLE_SECONDS:
                return False
            self.start_confirm(after_motion=True)
            return True

        if self.state == CONFIRMING:
            if changed:                  # hand came back mid-check
                motion.set_anchor(cur)
                self.state, self.still_since = SETTLING, now
                return False
            return True

        # IDLE: anchor is the settled scene. One noisy frame is ignored.
        if changed:
            self.hot += 1
            if self.hot >= MOTION_MIN_FRAMES:
                self.hot = 0
                motion.set_anchor(cur)
                self.state = MOVING
            return False
        self.hot = 0
        if now - self.last_check >= HEARTBEAT_SECONDS:
            self.start_confirm(after_motion=False)
            return True
        return False


def state_signature(groups):
    """What matters for the stacking state - ignores small pixel jitter."""
    return tuple(sorted((tuple(normalize_label(l) for l in g['order']), g['is_stack'])
                        for g in groups))


class GatedTracker:
    """Collects YOLO results during a confirm window and commits one state."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.window = []                 # (signature, cubes, groups) this window
        self.stable_sig = None
        self.changed = False             # last commit changed YOLO's reading
        self.cubes, self.groups = [], []

    @property
    def ready(self):
        return self.stable_sig is not None

    def start(self):
        self.window = []

    def add(self, cubes, groups):
        """Feed one YOLO result. True once the window is decided."""
        self.window.append((state_signature(groups), cubes, groups))

        recent = [w[0] for w in self.window[-CONFIRM_FRAMES:]]
        if len(recent) == CONFIRM_FRAMES and len(set(recent)) == 1:
            self._commit(*self.window[-1], unsettled=False)
            return True

        if len(self.window) >= CONFIRM_MAX_FRAMES:
            sigs = [w[0] for w in self.window]
            best = max(set(sigs), key=sigs.count)
            pick = next(w for w in reversed(self.window) if w[0] == best)
            self._commit(*pick, unsettled=True)
            return True
        return False

    def _commit(self, sig, cubes, groups, unsettled):
        self.cubes, self.groups = cubes, groups
        self.changed = sig != self.stable_sig
        if self.changed:
            self.stable_sig = sig
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            note = "  (results disagreed - majority taken)" if unsettled else ""
            line = f"{ts}  detected: {short_summary(groups)}{note}"
            print(line)
            with open(self.log_path, 'a') as f:
                f.write(line + "\n")


class GatedMonitor:
    """Motion detector + gate + tracker + VLM. step() per frame, render() to draw."""

    def __init__(self, model, output_folder, log_path, vlm=None):
        self.model = model
        self.output_folder, self.log_path = output_folder, log_path
        self.motion = MotionDetector()
        self.gate = Gate()
        self.tracker = GatedTracker(log_path)
        self.vlm = vlm                   # VLMWorker or None
        self.settle_id = 0               # one per settled scene (image + JSON)
        self.local = None                # code-only result of the last no-anomaly settle
        self.show_info = False
        self.frames = 0
        self.yolo_runs = 0

    def step(self, frame, now):
        self.frames += 1
        run_yolo = self.gate.step(self.motion, frame, now)
        if not run_yolo:
            return
        if self.gate.confirm_frames == 0:
            self.tracker.start()
        cubes, groups = detect_and_group(self.model, frame)
        self.yolo_runs += 1
        self.gate.confirm_frames += 1
        if self.tracker.add(cubes, groups):
            after_motion = self.gate.after_motion
            self.gate.done(now)
            # A new settle after every real motion - even if YOLO's reading did
            # not change, because YOLO may have missed what was just placed.
            # Heartbeat re-checks only count if YOLO's reading changed.
            if after_motion or self.tracker.changed:
                self.new_settle(frame)

    def new_settle(self, frame, force_vlm=False):
        """Save images 1-3; ask the VLM only on an anomaly (image 4 + JSON follow)."""
        self.settle_id += 1
        base = "settle_" + datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
        folder = self.output_folder / 'settles' / base
        frame = frame.copy()
        groups = self.tracker.groups
        yolo_only = detect_cubes(self.model, frame, checks=False)  # plain YOLO
        images = save_stage_images(folder, frame, yolo_only, groups)
        print(f"Settle {self.settle_id}: saved settles/{base}/ images 1-3")
        job = {'frame': frame, 'groups': groups, 'yolo_only': yolo_only,
               'images': images, 'folder': folder, 'settle_id': self.settle_id,
               'crop': self.motion.area_px(frame) if self.motion.work_area else None}

        triggers = vlm_triggers(groups)
        use_vlm = self.vlm is not None and (force_vlm or triggers
                                            or not VLM_ON_ANOMALY_ONLY)
        if not use_vlm:
            # No anomaly (or VLM off): answer from YOLO + geometry right away.
            why = ("VLM off: " + (VLM_OFF_REASON or "--no-vlm") if self.vlm is None
                   else "no anomaly against the goal")
            res = code_result(groups, why)
            res['settle_id'] = self.settle_id
            self.local = res
            print(format_vlm_report(res))
            finish_settle(folder, self.settle_id, frame, groups, yolo_only, images,
                          res, log_path=self.log_path)
            return
        self.vlm.submit(job)
        print(f"VLM: anomaly - {'; '.join(triggers) or 'forced check (v key)'}")

    def ask_vlm(self, frame):
        """'v' key: treat the current scene as a new settle."""
        if self.vlm is None:
            print("VLM is off")
            return
        if self.gate.state != IDLE or not self.tracker.ready:
            print("VLM: scene not settled yet - try again in a moment")
            return
        self.new_settle(frame, force_vlm=True)

    def recheck(self):
        self.gate.start_confirm(after_motion=True)

    # ---- what the worker sees ----

    def instructions(self):
        """(header, colour, problem, steps, notes, dim) for the panel."""
        if not self.tracker.ready:
            return "STARTING", COLOR_IDLE, "Looking at the work area...", [], [], False
        current = self.current_vlm()
        if current is not None:
            return (*instructions_from_result(current), False)
        if self.vlm is None:
            return "CHECKING THE STACK...", COLOR_IDLE, "", [], [], False

        older = [r for r in (self.vlm.latest(), self.local) if r is not None]
        previous = max(older, key=lambda r: r['settle_id']) if older else None
        if previous is not None and not previous['error'] and previous['steps']:
            # New placement not answered yet: keep the old list readable, greyed.
            return ("CHECKING YOUR MOVE...", COLOR_IDLE,
                    "Previous instructions:", previous['steps'], [], True)
        return "CHECKING THE STACK...", COLOR_IDLE, "", [], [], False

    def footer(self):
        state = self.gate.state
        if state == MOVING:
            return "Placing - waiting for your hand to leave", COLOR_WARN
        if state == SETTLING:
            return "Settling...", COLOR_WARN
        if state == CONFIRMING:
            return "Reading the cubes...", COLOR_IDLE
        if self.vlm is not None and self.vlm.thinking():
            return f"{self.vlm.client.model} is checking...", (200, 80, 160)
        return "Ready", COLOR_OK

    def render(self, frame, fps=0.0, paused=False):
        vis = frame.copy()
        state = self.gate.state

        # Boxes only for a settled, confirmed scene - hidden while the hand moves.
        if state in (IDLE, CONFIRMING) and self.tracker.ready:
            # the VLM's final reading once it has answered for this settle
            draw_final(vis, self.tracker.groups, self.current_vlm())

        header, colour, problem, steps, notes, dim = self.instructions()
        footer, footer_colour = self.footer()
        if paused:
            footer += "  [PAUSED]"
        draw_instruction_panel(vis, header, colour, problem, steps, footer,
                               footer_colour, dim,
                               side=choose_panel_side(self.tracker.groups, vis.shape[1]),
                               notes=notes)

        if self.show_info:
            if self.motion.work_area is not None:
                x1, y1, x2, y2 = self.motion.area_px(frame)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)
            pct = 100.0 * self.yolo_runs / max(1, self.frames)
            info = (f"{fps:.0f} FPS | {state} | motion {100 * self.motion.level:.1f}% | "
                    f"YOLO {self.yolo_runs}/{self.frames} ({pct:.1f}%) | "
                    f"settle {self.settle_id}"
                    + (f" | VLM {'busy' if self.vlm.thinking() else 'idle'}"
                       if self.vlm else ""))
            y = vis.shape[0] - 10
            cv2.putText(vis, info, (10, y), FONT, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, info, (10, y), FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def current_vlm(self):
        """Result for the current settle: code-only (no anomaly) or the VLM's."""
        if not self.tracker.ready:
            return None
        if self.local is not None and self.local['settle_id'] == self.settle_id:
            return self.local
        return self.vlm.latest(self.settle_id) if self.vlm is not None else None


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Goal (target stack) from an image
# --------------------------------------------------------------------------


def set_goal(goal, source):
    global EXPECTED_ORDER, GOAL_SOURCE
    EXPECTED_ORDER = [normalize_label(c) for c in goal]
    GOAL_SOURCE = source
    print(f"GOAL (bottom -> top): {' > '.join(EXPECTED_ORDER)}   [{source}]")


def read_goal(model, frame, vlm_client=None):
    """
    Read the target stack from a photo of it: YOLO + geometry, then the VLM
    (raw image) if available. The goal is the tallest stack in the reading.
    Returns dict(goal, state, source, groups, entries) - goal None if no stack.
    """
    _, groups = detect_and_group(model, frame)
    state, source, vlm_boxes = detector_state(groups), 'YOLO + geometry', {}
    if vlm_client is not None:
        try:
            info = {'raw': []}
            seen, probs = vlm_look(vlm_client, frame, groups, None, info)
            if seen is not None and not probs:
                det = state
                state, source = seen, 'VLM'
                need = list((Counter(all_colours(seen))
                             - Counter(all_colours(det))).elements())
                if need and VLM_GROUND:
                    vlm_boxes = vlm_ground(vlm_client, frame, need, seen, None, info)
            elif probs:
                print(f"Goal: VLM reading rejected ({'; '.join(probs)}) - using YOLO")
        except Exception as exc:
            print(f"Goal: VLM failed ({exc}) - using YOLO")
    stacks = [s for s in state['stacks'] if len(s) >= MIN_STACK_SIZE]
    goal = max(stacks, key=len) if stacks else None
    entries = []
    if goal is not None:
        entries, _, _ = final_boxes(groups, {'stacks': [goal], 'table': []},
                                    frame, vlm_boxes)
    return {'goal': goal, 'state': state, 'source': source,
            'groups': groups, 'entries': entries}


def goal_preview(frame, info, footer, footer_colour=COLOR_IDLE):
    """Frame with the goal stack boxed and a panel describing it."""
    img = draw_entries(frame.copy(), info['entries'])
    if info['goal']:
        problem = f"Goal, bottom to top: {' > '.join(info['goal'])}"
        steps = [c.upper() + (" (bottom, on the table)" if i == 0 else "")
                 for i, c in enumerate(info['goal'])]
        header, colour = "GOAL FOUND", COLOR_OK
    else:
        problem = (f"No stack found ({state_summary(info['state'])}). Build the "
                   f"target stack in view and capture again.")
        steps, header, colour = [], "NO STACK FOUND", COLOR_BAD
    draw_instruction_panel(img, header, colour, problem, steps,
                           footer, footer_colour, side='left',
                           notes=[f"read by: {info['source']}"])
    return img


def save_goal(output_folder, frame, info, source):
    folder = output_folder / 'goal' / ("goal_" + datetime.now().strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True, exist_ok=True)
    save_jpg(folder / 'goal_raw.jpg', frame)
    save_jpg(folder / 'goal_annotated.jpg', goal_preview(frame, info, source))
    (folder / 'goal.json').write_text(json.dumps({
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'goal_bottom_to_top': info['goal'], 'source': source,
        'read_by': info['source'], 'full_reading': info['state'],
        'boxes_bottom_to_top': info['entries']}, indent=2))
    print(f"Goal saved in {folder}")


def goal_from_image(model, path, vlm_client, output_folder):
    frame = cv2.imread(str(path))
    if frame is None:
        sys.exit(f"Could not read goal image: {path}")
    if RESIZE is not None:
        frame = cv2.resize(frame, RESIZE)
    info = read_goal(model, frame, vlm_client)
    if not info['goal']:
        sys.exit(f"No stack found in the goal image {path} "
                 f"(reading: {state_summary(info['state'])})")
    set_goal(info['goal'], f"image {Path(path).name}")
    save_goal(output_folder, frame, info, GOAL_SOURCE)


def capture_goal_live(camera, model, vlm_client, output_folder, win):
    """
    Live: the worker builds the target stack, presses SPACE, checks the
    reading and presses ENTER to accept (SPACE to capture again).
    Returns True when a goal is set, False if the user quit.
    """
    print("SET THE GOAL: build the target stack in view, press SPACE to capture it")
    frame, preview, info = None, None, None
    while True:
        if preview is None:
            new = camera.read()
            if new is not None:
                if RESIZE is not None and (new.shape[1], new.shape[0]) != tuple(RESIZE):
                    new = cv2.resize(new, RESIZE)
                frame = new
            if frame is None:
                continue
            view = frame.copy()
            draw_instruction_panel(
                view, "SET THE GOAL", COLOR_WARN,
                "Build the TARGET stack (the correct order) where the camera "
                "can see it, then press SPACE to capture it.", [],
                "SPACE capture  |  q quit", COLOR_IDLE, side='left')
        else:
            view = preview
        cv2.imshow(win, view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            return False
        if preview is None and key == 32:                 # SPACE: capture
            shot = frame.copy()
            busy = shot.copy()
            draw_instruction_panel(busy, "READING THE GOAL...", COLOR_IDLE,
                                   "Checking the stack (the first VLM call also "
                                   "loads the model).", [], "please wait",
                                   COLOR_IDLE, side='left')
            cv2.imshow(win, busy)
            cv2.waitKey(1)
            info = read_goal(model, shot, vlm_client)
            hint = ("ENTER accept  |  SPACE capture again  |  q quit" if info['goal']
                    else "SPACE capture again  |  q quit")
            preview = goal_preview(shot, info, hint)
            print(f"Goal reading: {state_summary(info['state'])} [{info['source']}]")
        elif preview is not None and key == 32:           # SPACE: retry
            preview = None
        elif preview is not None and key in (13, 10) and info['goal']:   # ENTER
            set_goal(info['goal'], 'captured live')
            save_goal(output_folder, shot, info, GOAL_SOURCE)
            return True


def run_live(model, camera, output_folder, vlm_client=None):
    win = "Cube stacking assistant"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if not EXPECTED_ORDER and not capture_goal_live(camera, model, vlm_client,
                                                    output_folder, win):
        camera.release()
        cv2.destroyAllWindows()
        return

    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_folder / f"stack_log_vlm_{session}.txt"
    log_path.write_text(
        f"Session {session} (motion-gated, VLM: "
        f"{vlm_client.model if vlm_client else 'off'}"
        f"{', planner ' + vlm_client.planner if vlm_client else ''})\n"
        f"Goal (bottom -> top): {' -> '.join(EXPECTED_ORDER)}\n\n")
    vlm = VLMWorker(vlm_client, output_folder, log_path) if vlm_client else None
    monitor = GatedMonitor(model, output_folder, log_path, vlm)

    paused = False
    fps = 0.0
    t_prev = time.time()
    frame = None
    window_seen = False

    print(f"Logging to {log_path}")
    print(f"Every settle -> {output_folder}/settles/settle_<time>/ (4 images + result.json)")
    print("Keys: q quit | s snapshot | p pause | i info | d debug | "
          "a work area | r re-check" + (" | v ask VLM" if vlm else ""))

    try:
        while True:
            if not paused:
                new = camera.read()
                if new is None:
                    continue
                if RESIZE is not None and (new.shape[1], new.shape[0]) != tuple(RESIZE):
                    new = cv2.resize(new, RESIZE)
                frame = new

                now = time.time()
                monitor.step(frame, now)
                fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))
                t_prev = now

            if frame is None:
                continue

            cv2.imshow(win, monitor.render(frame, fps, paused))

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            if key == ord('p'):
                paused = not paused
            if key == ord('i'):
                monitor.show_info = not monitor.show_info
            if key == ord('r'):
                monitor.recheck()
                print("Re-check requested")
            if key == ord('v'):
                monitor.ask_vlm(frame)
            if key == ord('d'):
                print(f"state={monitor.gate.state} motion={100 * monitor.motion.level:.2f}% "
                      f"(threshold {100 * MOTION_AREA_FRAC:.1f}%)")
                print_debug(monitor.tracker.cubes, model, frame)
            if key == ord('a'):
                x, y, w, h = cv2.selectROI(win, frame, showCrosshair=False)
                fh, fw = frame.shape[:2]
                monitor.motion.work_area = (None if w == 0 or h == 0 else
                                            (x / fw, y / fh, (x + w) / fw, (y + h) / fh))
                monitor.motion.reset()
                print(f"Work area: {monitor.motion.work_area or 'full frame'}")
            if key == ord('s'):
                stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                img_path = output_folder / f"snapshot_{stamp}.jpg"
                txt_path = output_folder / f"snapshot_{stamp}_stack.txt"
                cv2.imwrite(str(img_path), monitor.render(frame, fps, paused))
                report = build_report(img_path.name, monitor.tracker.groups)
                res = monitor.current_vlm()
                if res is not None:
                    report += "\n" + format_vlm_report(res)
                txt_path.write_text(report)
                print(f"Saved {img_path} and {txt_path}")
            # Window closed with the X button. Qt reports "not visible" for the
            # first frames before the window is mapped, so only trust it after
            # the window has been seen visible once.
            visible = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1
            if visible:
                window_seen = True
            elif window_seen:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if vlm is not None:
            vlm.drain()
        print(f"YOLO ran on {monitor.yolo_runs} of {monitor.frames} frames "
              f"({100.0 * monitor.yolo_runs / max(1, monitor.frames):.1f}%), "
              f"{monitor.settle_id} settle(s) saved")


def run_single_image(model, image_path, output_folder, vlm_client=None):
    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"Could not read image: {image_path}")
    if RESIZE is not None:
        image = cv2.resize(image, RESIZE)

    _, groups = detect_and_group(model, image)
    yolo_only = detect_cubes(model, image, checks=False)
    report = build_report(image_path.name, groups)
    print(report)

    folder = output_folder / 'settles' / image_path.stem
    images = save_stage_images(folder, image, yolo_only, groups)

    triggers = vlm_triggers(groups)
    if vlm_client is not None and (triggers or not VLM_ON_ANOMALY_ONLY):
        print(f"Anomaly ({'; '.join(triggers) or 'check every image'}) - "
              f"asking {vlm_client.model} ...")
        res = run_vlm_check(vlm_client, image, groups)
    else:
        res = code_result(groups, "no anomaly against the goal" if vlm_client
                          else "VLM off: " + (VLM_OFF_REASON or "--no-vlm"))
    res['settle_id'] = 1
    print(format_vlm_report(res))
    report += "\n" + format_vlm_report(res)

    json_path = finish_settle(folder, 1, image, groups, yolo_only, images, res)
    (folder / "report.txt").write_text(report)
    print(f"Saved {folder}/ (4 images, result.json, report.txt)")


def main():
    parser = argparse.ArgumentParser(
        description="Cube stacking assistant: YOLO detects, a VLM checks and instructs")
    parser.add_argument('--source', choices=['oak', 'webcam'], default='oak',
                        help="camera source for live mode (default: oak)")
    parser.add_argument('--cam', type=int, default=0,
                        help="OpenCV camera index when --source webcam")
    parser.add_argument('--image', type=str, default=None,
                        help="process a single image instead of live mode")
    parser.add_argument('--no-vlm', action='store_true',
                        help="disable the VLM (detections only, no instructions)")
    parser.add_argument('--vlm-model', default=VLM_MODEL,
                        help=f"Ollama VISION model for the LOOK step (default: {VLM_MODEL})")
    parser.add_argument('--planner-model', default=VLM_PLANNER_MODEL,
                        help=f"text model for the planning step (default: "
                             f"{VLM_PLANNER_MODEL}); same as --vlm-model if empty")
    parser.add_argument('--vlm-backend', choices=['api', 'cli'], default=VLM_BACKEND,
                        help="'api' = Ollama REST server, 'cli' = `ollama run` subprocess")
    parser.add_argument('--ollama-url', default=OLLAMA_URL,
                        help=f"Ollama server for the api backend (default: {OLLAMA_URL})")
    parser.add_argument('--goal-image', default=None,
                        help="photo of the TARGET stack; its order (bottom -> top) "
                             "becomes the goal. Default: capture it live at start")
    parser.add_argument('--goal', default=None,
                        help="goal order bottom -> top, e.g. red,green,black,white "
                             "(skips the goal photo)")
    args = parser.parse_args()

    global VLM_OFF_REASON, MODEL_COLOURS
    vlm_client = None
    if VLM_ENABLED and not args.no_vlm:
        vlm_client = OllamaVLM(args.vlm_model, args.vlm_backend, args.ollama_url,
                                planner=args.planner_model or None)
        ok, msg = vlm_client.check()
        if ok:
            print(msg)
        else:
            VLM_OFF_REASON = msg
            print("\n" + "!" * 70 + f"\nVLM NOT STARTED: {msg}\n"
                  "Continuing with YOLO + geometry only.\n" + "!" * 70 + "\n")
            vlm_client = None

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_folder = (script_dir / OUTPUT_FOLDER).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(str((script_dir / MODEL_PATH).resolve()))
    MODEL_COLOURS = list(dict.fromkeys(normalize_label(n) for n in model.names.values()))

    # ---- goal: --goal, --goal-image, or captured live (default) ----
    if args.goal:
        goal = [normalize_label(c) for c in args.goal.split(',') if c.strip()]
        unknown = [c for c in goal if c not in MODEL_COLOURS]
        if unknown:
            sys.exit(f"--goal has colours the model does not know: {unknown}. "
                     f"Known: {MODEL_COLOURS}")
        set_goal(goal, '--goal argument')
    elif args.goal_image:
        goal_from_image(model, Path(args.goal_image).resolve(), vlm_client, output_folder)
    elif args.image:
        sys.exit("With --image, give the goal too: --goal-image FILE or --goal a,b,c")

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            sys.exit(f"Image not found: {image_path}")
        run_single_image(model, image_path, output_folder, vlm_client)
        return

    size = tuple(RESIZE) if RESIZE is not None else (1280, 720)
    if args.source == 'oak':
        camera = OakCamera(size, CAMERA_FPS)
    else:
        camera = WebcamCamera(args.cam, size, CAMERA_FPS)

    run_live(model, camera, output_folder, vlm_client)


if __name__ == '__main__':
    main()

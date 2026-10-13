#!/usr/bin/env python3
"""
LIVE cube stacking monitor - MOTION-GATED version.

Same detection, colour checks, hidden-cube inference and guidance as
ko_yolo_stack_live.py, but YOLO does not run on every frame. A cheap motion
detector watches the work area instead:

    IDLE        scene still, last result shown, no YOLO (except a heartbeat)
    MOVING      hand / cube moving, no YOLO, banner says "placing"
    SETTLING    motion stopped, waiting SETTLE_SECONDS to be sure
    CONFIRMING  YOLO runs until CONFIRM_FRAMES results agree, then -> IDLE

So guidance is only ever given on a finished placement, and a cube held in
the hand is never evaluated.


Detects coloured cubes with YOLO on every camera frame, separates genuine stacks
from loose cubes on the table, reads each stack's bottom -> top order, validates
it against the expected sequence and tells the worker what to do next.

Each detection box is drawn in the cube's own colour (red cube -> red box).
Problems are shown with an extra outer frame and a tag in the label:
    magenta frame + [WRONG] / [EXTRA]  -> cube is in the wrong place
    yellow frame  + [CHECK]            -> gap / spread warning for that stack
A '*' after a label means the colour check overrode YOLO's class.
A '?' and a dashed box mean the cube was not detected by YOLO but inferred
from the colour directly below a stack (see INFER_HIDDEN).

Sources:
    python ko_yolo_stack_live_gated.py                        # OAK-D Pro (default)
    python ko_yolo_stack_live_gated.py --source webcam --cam 0
    python ko_yolo_stack_live_gated.py --image path/to/img.jpg   # old single-image mode

Keys (live mode):
    q / ESC : quit
    s       : save snapshot (annotated image + full text report)
    p       : pause / resume
    d       : print every detection with its HSV colour values (for tuning)
    a       : draw the work area for motion detection (ENTER to confirm,
              c to cancel; an empty box resets to the full frame)
    r       : force a re-check now
"""

import argparse
import os
import sys
import time
import zlib
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

EXPECTED_ORDER = ['red', 'green', 'black', 'white']   # bottom -> top

MIN_STACK_SIZE = 2

SAME_STACK_X_FRAC = 0.6      # centre offset, as a fraction of mean cube width
SIZE_RATIO_TOL = 0.25        # box widths must agree within 25 %

GAP_MULTIPLE_WARN = 1.6      # gap > 1.6 * the typical gap -> cube probably missing
X_SPREAD_WARN_FRAC = 0.8     # members spread wider than this -> stacks may be merged

LABEL_NOISE_TOKENS = ('cube', 'cubes', 'block', 'blocks', 'box', 'boxes')

# Live mode: a new stacking state is only accepted (and logged) after it has
# been seen in this many consecutive frames. Stops guidance from flickering
# when a detection drops out for a frame or a hand passes over the stack.
STABLE_FRAMES = 30

# ---- Motion gating (this version only) ----
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

    base = normalize_label(EXPECTED_ORDER[0])
    h_img = image.shape[0]
    inferred = []

    for group in groups:
        bottom = group['members'][0]
        if normalize_label(bottom['label']) == base:
            continue

        step = float(np.median(group['gaps'])) if group['gaps'] else bottom['h']
        cur = bottom
        for _ in range(len(EXPECTED_ORDER) - len(group['members'])):
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


def detect_and_group(model, image):
    """Detection -> grouping -> hidden-cube inference -> regrouping."""
    cubes = detect_cubes(model, image)
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
    return float(np.median([m['w'] for m in cluster]))


def _can_join(cube, cluster):
    """Same horizontal column AND similar apparent size."""
    ref_w = _cluster_width(cluster)
    if ref_w <= 0:
        return False, None

    if abs(cube['w'] - ref_w) / ref_w > SIZE_RATIO_TOL:
        return False, None

    centre = float(np.mean([m['cx'] for m in cluster]))
    dist = abs(cube['cx'] - centre)
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
    mark_build_site(groups)
    return groups


def validate_order(order, expected=EXPECTED_ORDER):
    """Returns (status, first_bad_index)."""
    norm = [normalize_label(c) for c in order]
    exp = [normalize_label(c) for c in expected]

    if len(norm) > len(exp):
        return 'TOO_MANY_CUBES', len(exp)

    for i, label in enumerate(norm):
        if label != exp[i]:
            return 'WRONG_ORDER', i

    if len(norm) < len(exp):
        return 'INCOMPLETE', None
    return 'OK', None


def find_build_site(groups):
    """The group whose BOTTOM cube is the expected base colour."""
    base = normalize_label(EXPECTED_ORDER[0])
    candidates = [g for g in groups if normalize_label(g['order'][0]) == base]
    if not candidates:
        return None
    return max(candidates, key=lambda g: len(g['members']))


def mark_build_site(groups):
    site = find_build_site(groups)
    for group in groups:
        group['is_site'] = group is site
    return site


def site_status(groups):
    """Validation status of the build site, or None if no base is placed."""
    site = next((g for g in groups if g.get('is_site')), None)
    if site is None:
        return None
    return validate_order(site['order'])[0]


def build_guidance(groups):
    """One line telling the worker what to do next."""
    base = normalize_label(EXPECTED_ORDER[0])

    if not groups:
        return ("no cubes detected - this is a vision problem, not a stacking "
                "error (check lighting, camera angle, or lower CONF)")

    site = next((g for g in groups if g.get('is_site')), None)

    if site is None:
        stacks = [g for g in groups if g['is_stack']]
        if stacks:
            s = max(stacks, key=lambda g: len(g['members']))
            bottom = normalize_label(s['order'][0])
            return (f"stack has {bottom} at the bottom and no {base} base was found - "
                    f"if a {base} cube is under it, it is not being seen; "
                    f"otherwise rebuild starting with {base}")
        on_table = [g['members'][0]['label'] for g in groups if not g['is_stack']]
        extra = f" (on the table: {', '.join(on_table)})" if on_table else ""
        return f"no {base} cube placed yet - put a {base} cube down as the base{extra}"

    status, bad_idx = validate_order(site['order'])
    top = site['members'][-1]

    if status == 'OK':
        return f"stack complete - all {len(EXPECTED_ORDER)} cubes in the correct order"
    if status == 'WRONG_ORDER':
        wrong = site['order'][bad_idx]
        return (f"remove '{wrong}' at position {bad_idx} - "
                f"'{EXPECTED_ORDER[bad_idx]}' belongs there")
    if status == 'TOO_MANY_CUBES':
        return f"too many cubes - remove '{top['label']}' from the top"

    nxt = EXPECTED_ORDER[len(site['order'])]
    msg = (f"place the {nxt} cube on top of '{top['label']}' "
           f"at ({top['cx']:.0f},{top['cy']:.0f})")

    for group in groups:
        if group['is_stack'] or group.get('is_site'):
            continue
        cube = group['members'][0]
        if normalize_label(cube['label']) == normalize_label(nxt):
            msg += f" - it is on the table at ({cube['cx']:.0f},{cube['cy']:.0f})"
            break
    return msg


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

COLOR_OK = (0, 200, 0)
COLOR_BAD = (0, 0, 255)
COLOR_WARN = (0, 165, 255)
COLOR_LOOSE = (255, 190, 0)
COLOR_IDLE = (90, 90, 90)
# Status frames drawn around a cube's own box. Chosen to stand apart from
# the cube colours so a red cube in the wrong place is still clearly flagged.
FLAG_BAD = (255, 0, 255)      # magenta
FLAG_WARN = (0, 255, 255)     # yellow
FONT = cv2.FONT_HERSHEY_SIMPLEX


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


def draw_cube_box(image, cube, text, flag=None):
    """Box + filled label in the cube's own colour, optional status frame."""
    color = cube_color(cube['label'])
    edge = contrast_color(color)
    p1, p2 = (cube['x1'], cube['y1']), (cube['x2'], cube['y2'])

    if cube.get('inferred'):
        # Dashed: this cube was inferred from colour, not detected by YOLO.
        dashed_rect(image, p1, p2, edge, 4)
        dashed_rect(image, p1, p2, color, 2)
    else:
        # Thin contrasting halo keeps black boxes visible on dark scenes and
        # white boxes visible on a white table.
        cv2.rectangle(image, p1, p2, edge, 4)
        cv2.rectangle(image, p1, p2, color, 2)

    if flag is not None:
        cv2.rectangle(image, (p1[0] - 5, p1[1] - 5), (p2[0] + 5, p2[1] + 5), flag, 2)

    scale, thick = 0.45, 1
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    lx = cube['x1']
    ly2 = cube['y1'] - (7 if flag is not None else 2)
    ly1 = ly2 - th - base - 4
    if ly1 < 0:                              # no room above: put label inside box
        ly1 = cube['y1'] + 2
        ly2 = ly1 + th + base + 4
    cv2.rectangle(image, (lx, ly1), (lx + tw + 6, ly2), color, -1)
    cv2.rectangle(image, (lx, ly1), (lx + tw + 6, ly2), edge, 1)
    cv2.putText(image, text, (lx + 3, ly2 - base - 2), FONT, scale, edge, thick,
                cv2.LINE_AA)


def annotate(image, groups, show_guidance=True):
    stack_no = 0
    for group in groups:
        if group['is_stack']:
            stack_no += 1
            status, bad_idx = validate_order(group['order'])
            tag = f"S{stack_no}"
        else:
            status, bad_idx = None, None
            tag = "B" if group.get('is_site') else "L"

        for cube in group['members']:
            pos = cube['stack_pos']

            flag, flag_txt = None, ""
            if group['is_stack']:
                if status == 'WRONG_ORDER' and pos == bad_idx:
                    flag, flag_txt = FLAG_BAD, " [WRONG]"
                elif status == 'TOO_MANY_CUBES' and pos >= bad_idx:
                    flag, flag_txt = FLAG_BAD, " [EXTRA]"
                elif not group['gaps_ok'] or not group['x_spread_ok']:
                    flag, flag_txt = FLAG_WARN, " [CHECK]"

            name = (cube['label'] + ('*' if cube.get('recoloured') else '')
                    + ('?' if cube.get('inferred') else ''))
            if group['is_stack']:
                text = f"{tag}#{pos} {name}"
            elif group.get('is_site'):
                text = f"{name} (base)"
            else:
                text = f"{name} (loose)"
            if SHOW_ANGLE:
                text += f" {cube['angle']}deg"
            text += flag_txt

            draw_cube_box(image, cube, text, flag)

    # Summary panel
    y = 20
    stack_no = 0
    loose = []
    for group in groups:
        if not group['is_stack']:
            if group.get('is_site'):
                cv2.putText(image, f"BASE [{group['order'][0]}]", (8, y),
                            FONT, 0.5, COLOR_OK, 2)
                y += 20
            else:
                loose.append(group['members'][0]['label'])
            continue
        stack_no += 1
        status, _ = validate_order(group['order'])
        color = COLOR_OK if status in ('OK', 'INCOMPLETE') else COLOR_BAD
        cv2.putText(image, f"S{stack_no} [{' -> '.join(group['order'])}] {status}",
                    (8, y), FONT, 0.5, color, 2)
        y += 20

    if loose:
        cv2.putText(image, f"Loose: {', '.join(loose)}", (8, y),
                    FONT, 0.5, COLOR_LOOSE, 2)
        y += 20

    if show_guidance:
        cv2.putText(image, f"NEXT: {build_guidance(groups)}", (8, y),
                    FONT, 0.45, COLOR_WARN, 2)
    return image


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


def draw_banner(image, text, color, scale=0.55, thickness=2):
    """Coloured guidance banner across the bottom of the frame."""
    h, w = image.shape[:2]
    lines = wrap_text(text, w - 16, scale, thickness)
    line_h = int(26 * scale / 0.55)
    top = h - line_h * len(lines) - 10
    cv2.rectangle(image, (0, top), (w, h), color, -1)
    for i, line in enumerate(lines):
        cv2.putText(image, line, (8, top + line_h * (i + 1)),
                    FONT, scale, (255, 255, 255), thickness)


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


def build_report(image_name, groups):
    stacks = [g for g in groups if g['is_stack']]
    loose = [g for g in groups if not g['is_stack']]

    lines = [f"Image: {image_name}",
             f"Expected order (bottom -> top): {' -> '.join(EXPECTED_ORDER)}",
             f"Stacks found: {len(stacks)}",
             f"Loose cubes on table: {len(loose)}", ""]

    for s_idx, group in enumerate(stacks, 1):
        status, bad_idx = validate_order(group['order'])
        lines.append(f"Stack {s_idx}: {len(group['members'])} cubes")
        for cube in group['members']:
            lines.append(f"  pos {cube['stack_pos']}: {cube_line(cube)}")
        lines.append(f"  order  : {' -> '.join(group['order'])}")
        lines.append(f"  status : {status}")
        lines.append(f"  gaps   : {group['gaps']} (typical {group['typical_gap']}), "
                     f"x-spread {group['x_spread']}")
        if status == 'WRONG_ORDER':
            lines.append(f"  problem: position {bad_idx} is "
                         f"'{group['order'][bad_idx]}', expected "
                         f"'{EXPECTED_ORDER[bad_idx]}'")
        elif status == 'TOO_MANY_CUBES':
            lines.append(f"  problem: more cubes than the expected {len(EXPECTED_ORDER)}")
        elif status == 'INCOMPLETE':
            missing = EXPECTED_ORDER[len(group['order']):]
            if len(missing) > 1:
                lines.append(f"  next   : add {missing[0]} (then {', '.join(missing[1:])})")
            else:
                lines.append(f"  next   : add {missing[0]}")
        if not group['gaps_ok']:
            lines.append(f"  warning: oversized gap(s) {group['big_gaps']} vs typical "
                         f"{group['typical_gap']} - a cube is occluded, missed, "
                         f"or two stacks were merged here")
        if not group['x_spread_ok']:
            lines.append(f"  warning: members span {group['x_spread']}px horizontally "
                         f"- this may be two adjacent stacks read as one")
        lines.append("")

    if loose:
        lines.append("Single cubes (not stacked):")
        for group in loose:
            cube = group['members'][0]
            role = 'BASE - stack starts here' if group.get('is_site') else 'WAITING'
            lines.append(f"  {cube_line(cube)}  [{role}]")
        lines.append("")

    lines.append(f"guidance: {build_guidance(groups)}")
    return "\n".join(lines)


def short_summary(groups):
    """Compact one-line state description for the event log."""
    parts = []
    stack_no = 0
    for group in groups:
        if group['is_stack']:
            stack_no += 1
            status, _ = validate_order(group['order'])
            parts.append(f"S{stack_no}[{'>'.join(group['order'])}]={status}")
        elif group.get('is_site'):
            parts.append(f"BASE[{group['order'][0]}]")
        else:
            parts.append(f"loose:{group['order'][0]}")
    return ' | '.join(parts) if parts else 'no cubes'


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
# Live-mode helpers
# --------------------------------------------------------------------------


def state_signature(groups):
    """What matters for the stacking state - ignores small pixel jitter."""
    sig = []
    for g in groups:
        order = tuple(normalize_label(lbl) for lbl in g['order'])
        sig.append((order, g['is_stack'], bool(g.get('is_site'))))
    return tuple(sorted(sig))


class StateTracker:
    """Accepts a new state only after STABLE_FRAMES identical frames."""

    def __init__(self, n_frames, log_path):
        self.n = n_frames
        self.candidate = None
        self.count = 0
        self.stable_sig = None
        self.guidance = "detecting cubes..."
        self.status = None
        self.log_path = log_path

    @property
    def ready(self):
        return self.stable_sig is not None

    def update(self, groups):
        sig = state_signature(groups)
        if sig == self.candidate:
            self.count += 1
        else:
            self.candidate, self.count = sig, 1

        if self.count < self.n:
            return  # not stable yet - keep showing the last accepted guidance

        # Current frame matches a stable state: refresh text (live coordinates).
        self.guidance = build_guidance(groups)
        self.status = site_status(groups)

        if sig != self.stable_sig:
            self.stable_sig = sig
            self._log(groups)

    def _log(self, groups):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"{ts}  {short_summary(groups)}  ->  {self.guidance}"
        print(line)
        with open(self.log_path, 'a') as f:
            f.write(line + "\n")

    def banner_color(self):
        if not self.ready:
            return COLOR_IDLE
        if self.status == 'OK':
            return COLOR_OK
        if self.status in ('WRONG_ORDER', 'TOO_MANY_CUBES'):
            return COLOR_BAD
        return COLOR_WARN


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
# Modes
# --------------------------------------------------------------------------


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
        self.still_since = None
        self.last_check = 0.0
        self.confirm_frames = 0
        self.hot = 0                     # IDLE: consecutive changed frames

    def start_confirm(self):
        self.state, self.confirm_frames = CONFIRMING, 0

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
            self.start_confirm()
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
            self.start_confirm()
            return True
        return False


class GatedTracker:
    """Collects YOLO results during a confirm window and commits one state."""

    def __init__(self, log_path):
        self.log_path = log_path
        self.window = []                 # (signature, cubes, groups) this window
        self.stable_sig = None
        self.cubes, self.groups = [], []
        self.guidance = "detecting cubes..."
        self.status = None

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
        self.guidance = build_guidance(groups)
        self.status = site_status(groups)
        if sig != self.stable_sig:
            self.stable_sig = sig
            ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            note = "  (results disagreed - majority taken)" if unsettled else ""
            line = f"{ts}  {short_summary(groups)}  ->  {self.guidance}{note}"
            print(line)
            with open(self.log_path, 'a') as f:
                f.write(line + "\n")

    def banner_color(self):
        if not self.ready:
            return COLOR_IDLE
        if self.status == 'OK':
            return COLOR_OK
        if self.status in ('WRONG_ORDER', 'TOO_MANY_CUBES'):
            return COLOR_BAD
        return COLOR_WARN


class GatedMonitor:
    """Motion detector + gate + tracker. step() per frame, render() to draw."""

    def __init__(self, model, log_path):
        self.model = model
        self.motion = MotionDetector()
        self.gate = Gate()
        self.tracker = GatedTracker(log_path)
        self.frames = 0
        self.yolo_runs = 0

    def step(self, frame, now):
        self.frames += 1
        if not self.gate.step(self.motion, frame, now):
            return
        if self.gate.confirm_frames == 0:
            self.tracker.start()
        cubes, groups = detect_and_group(self.model, frame)
        self.yolo_runs += 1
        self.gate.confirm_frames += 1
        if self.tracker.add(cubes, groups):
            self.gate.done(now)

    def recheck(self):
        self.gate.start_confirm()

    def render(self, frame, fps=0.0, paused=False):
        vis = frame.copy()
        state = self.gate.state

        if state in (MOVING, SETTLING):
            # Boxes from before the hand came in may be stale - hide them.
            msg = ("PLACING - waiting for the hand to leave the work area"
                   if state == MOVING else "settling...")
            draw_banner(vis, msg, COLOR_IDLE)
        else:
            annotate(vis, self.tracker.groups, show_guidance=False)
            text = f"NEXT: {self.tracker.guidance}"
            if state == CONFIRMING and self.tracker.ready:
                text = f"(checking) {text}"
            draw_banner(vis, text, self.tracker.banner_color())

        if self.motion.work_area is not None:
            x1, y1, x2, y2 = self.motion.area_px(frame)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        pct = 100.0 * self.yolo_runs / max(1, self.frames)
        status = (f"{fps:.0f} FPS | {state} | motion {100 * self.motion.level:.1f}% | "
                  f"YOLO {self.yolo_runs}/{self.frames} ({pct:.1f}%)"
                  + ("  [PAUSED]" if paused else ""))
        (tw, _), _ = cv2.getTextSize(status, FONT, 0.5, 1)
        cv2.putText(vis, status, (vis.shape[1] - tw - 8, 20), FONT, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, status, (vis.shape[1] - tw - 8, 20), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        return vis


# --------------------------------------------------------------------------
# Modes
# --------------------------------------------------------------------------


def run_live(model, camera, output_folder):
    session = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_folder / f"stack_log_gated_{session}.txt"
    log_path.write_text(
        f"Session {session} (motion-gated)\nExpected order (bottom -> top): "
        f"{' -> '.join(EXPECTED_ORDER)}\n\n")
    monitor = GatedMonitor(model, log_path)

    win = "Cube stacking monitor (gated)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    paused = False
    fps = 0.0
    t_prev = time.time()
    frame = None

    print(f"Logging state changes to {log_path}")
    print("Keys: q = quit, s = snapshot, p = pause, d = debug, "
          "a = set work area, r = re-check now")

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
            if key == ord('r'):
                monitor.recheck()
                print("Re-check requested")
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
                txt_path.write_text(build_report(img_path.name, monitor.tracker.groups))
                print(f"Saved {img_path} and {txt_path}")
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print(f"YOLO ran on {monitor.yolo_runs} of {monitor.frames} frames "
              f"({100.0 * monitor.yolo_runs / max(1, monitor.frames):.1f}%)")


def run_single_image(model, image_path, output_folder):
    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"Could not read image: {image_path}")
    if RESIZE is not None:
        image = cv2.resize(image, RESIZE)

    _, groups = detect_and_group(model, image)
    report = build_report(image_path.name, groups)
    print(report)

    annotate(image, groups)
    out_img = output_folder / image_path.name
    out_txt = output_folder / f"{image_path.stem}_stack.txt"
    cv2.imwrite(str(out_img), image)
    out_txt.write_text(report)
    print(f"Saved {out_img}")
    print(f"Saved {out_txt}")


def main():
    parser = argparse.ArgumentParser(description="Motion-gated live cube stacking monitor")
    parser.add_argument('--source', choices=['oak', 'webcam'], default='oak',
                        help="camera source for live mode (default: oak)")
    parser.add_argument('--cam', type=int, default=0,
                        help="OpenCV camera index when --source webcam")
    parser.add_argument('--image', type=str, default=None,
                        help="process a single image instead of live mode")
    args = parser.parse_args()

    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    output_folder = (script_dir / OUTPUT_FOLDER).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    model = YOLO(str((script_dir / MODEL_PATH).resolve()))

    if args.image:
        image_path = Path(args.image).resolve()
        if not image_path.is_file():
            sys.exit(f"Image not found: {image_path}")
        run_single_image(model, image_path, output_folder)
        return

    size = tuple(RESIZE) if RESIZE is not None else (1280, 720)
    if args.source == 'oak':
        camera = OakCamera(size, CAMERA_FPS)
    else:
        camera = WebcamCamera(args.cam, size, CAMERA_FPS)

    run_live(model, camera, output_folder)


if __name__ == '__main__':
    main()

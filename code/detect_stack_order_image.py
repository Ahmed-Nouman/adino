#!/usr/bin/env python3
"""
Detect coloured cubes in ONE image, separate genuine stacks from loose cubes
lying on the table, read off each stack's bottom -> top order, and validate it
against the expected sequence.

Usage:
    python ko_yolo_stack_order.py                  # uses DEFAULT_IMAGE
    python ko_yolo_stack_order.py path/to/img.jpg
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = '../model/best.pt'
DEFAULT_IMAGE = '../test/stack.jpg'   # used when no CLI argument given
OUTPUT_FOLDER = '../result'

CONF = 0.8
RESIZE = (640, 480)                 # set to None to keep the native resolution
SHOW_ANGLE = False                  # True -> also compute per-cube orientation

EXPECTED_ORDER = ['red', 'green', 'black', 'white']   # bottom -> top

# A group of cubes is only called a stack once it has this many members.
# Anything smaller is reported as a loose cube on the table.
MIN_STACK_SIZE = 2

# Two cubes belong to the same stack only if their centres line up horizontally
# AND their boxes are a similar size. Apparent size falls off with distance, so
# a cube further back on the table is visibly smaller than one stacked directly
# above another - this is the closest 2D stand-in for depth.
SAME_STACK_X_FRAC = 0.6      # centre offset, as a fraction of mean cube width
SIZE_RATIO_TOL = 0.25        # box widths must agree within 25 %

GAP_MULTIPLE_WARN = 1.6      # gap > 1.6 * the typical gap -> cube probably missing
X_SPREAD_WARN_FRAC = 0.8     # members spread wider than this -> stacks may be merged

# Class names may be 'red_cube', 'Red Cube', 'red-block' etc. These tokens are
# stripped before comparing against EXPECTED_ORDER.
LABEL_NOISE_TOKENS = ('cube', 'cubes', 'block', 'blocks', 'box', 'boxes')

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


def detect_cubes(model, image):
    """Run YOLO and return a flat list of cube dicts."""
    h_img, w_img = image.shape[:2]
    cubes = []

    results = model(image, conf=CONF, verbose=False)
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            cls_id = int(box.cls[0])

            cube = {
                'label': model.names[cls_id],
                'conf': float(box.conf[0]),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                'cx': (x1 + x2) / 2.0,
                'cy': (y1 + y2) / 2.0,
                'w': float(x2 - x1),
                'h': float(y2 - y1),
                'angle': 0.0,
            }

            if SHOW_ANGLE:
                pad = 5
                roi = image[max(0, y1 - pad):min(h_img, y2 + pad),
                            max(0, x1 - pad):min(w_img, x2 + pad)]
                cube['angle'] = get_orientation_angle(roi)

            cubes.append(cube)

    return cubes


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
    """
    Cluster cubes into vertical columns.

    Returns a list of group dicts, ordered left to right:
        members      - cubes, index 0 = bottom
        is_stack     - True when the group has MIN_STACK_SIZE or more members
        order        - labels, bottom -> top
        gaps         - centre-to-centre vertical gaps
        gaps_ok      - False if a gap is far larger than the typical one
        x_spread_ok  - False if members are spread too wide to be one stack
    """
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

    # Calibrate expected spacing from the image itself. Under a tilted view the
    # centre-to-centre gap is well below the box height, so comparing gaps
    # against each other beats comparing them to a cube dimension.
    all_gaps = [g for g in (gap for grp in groups for gap in grp['gaps'])]
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
    """
    Compare a bottom -> top order against the expected sequence.
    Returns (status, first_bad_index); first_bad_index is None when not applicable.
    """
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
    """
    The group where the stack is being built: the one whose BOTTOM cube is the
    expected base colour. A lone red cube on the table already counts - it is a
    placed base, not a cube waiting to be placed.
    """
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


def build_guidance(groups):
    """One line telling the worker what to do next."""
    base = normalize_label(EXPECTED_ORDER[0])
    site = next((g for g in groups if g.get('is_site')), None)

    if site is None:
        on_table = [g['members'][0]['label'] for g in groups if not g['is_stack']]
        extra = f" (on the table: {', '.join(on_table)})" if on_table else ""
        return f"no {base} cube placed yet - put a {base} cube down as the base{extra}"

    status, bad_idx = validate_order(site['order'])
    top = site['members'][-1]

    if status == 'OK':
        return "stack complete - all four cubes in the correct order"
    if status == 'WRONG_ORDER':
        wrong = site['order'][bad_idx]
        return (f"remove '{wrong}' at position {bad_idx} - "
                f"'{EXPECTED_ORDER[bad_idx]}' belongs there")
    if status == 'TOO_MANY_CUBES':
        return f"too many cubes - remove '{top['label']}' from the top"

    # INCOMPLETE: the sequence so far is correct, say what comes next and where.
    nxt = EXPECTED_ORDER[len(site['order'])]
    msg = (f"place the {nxt} cube on top of '{top['label']}' "
           f"at ({top['cx']:.0f},{top['cy']:.0f})")

    # If that cube is already lying on the table, point to it.
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
COLOR_LOOSE = (255, 190, 0)     # cyan-blue: on the table, not part of a stack


def annotate(image, groups):
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
            if not group['is_stack']:
                color = COLOR_OK if group.get('is_site') else COLOR_LOOSE
            elif status == 'WRONG_ORDER' and pos == bad_idx:
                color = COLOR_BAD
            elif status == 'TOO_MANY_CUBES' and pos >= bad_idx:
                color = COLOR_BAD
            elif not group['gaps_ok'] or not group['x_spread_ok']:
                color = COLOR_WARN
            else:
                color = COLOR_OK

            cv2.rectangle(image, (cube['x1'], cube['y1']), (cube['x2'], cube['y2']), color, 2)

            if group['is_stack']:
                text = f"{tag}#{pos} {cube['label']}"
            elif group.get('is_site'):
                text = f"{cube['label']} (base)"
            else:
                text = f"{cube['label']} (loose)"
            if SHOW_ANGLE:
                text += f" {cube['angle']}deg"
            cv2.putText(image, text, (cube['x1'], max(12, cube['y1'] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    # Summary panel
    y = 20
    stack_no = 0
    loose = []
    for group in groups:
        if not group['is_stack']:
            if group.get('is_site'):
                cv2.putText(image, f"BASE [{group['order'][0]}]", (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_OK, 2)
                y += 20
            else:
                loose.append(group['members'][0]['label'])
            continue
        stack_no += 1
        status, _ = validate_order(group['order'])
        color = COLOR_OK if status in ('OK', 'INCOMPLETE') else COLOR_BAD
        cv2.putText(image, f"S{stack_no} [{' -> '.join(group['order'])}] {status}",
                    (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        y += 20

    if loose:
        cv2.putText(image, f"Loose: {', '.join(loose)}", (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_LOOSE, 2)
        y += 20

    cv2.putText(image, f"NEXT: {build_guidance(groups)}", (8, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_WARN, 2)
    return image


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
            lines.append(f"  pos {cube['stack_pos']}: {cube['label']:<12} "
                         f"conf={cube['conf']:.2f} "
                         f"centre=({cube['cx']:.0f},{cube['cy']:.0f}) "
                         f"box={cube['w']:.0f}x{cube['h']:.0f}")
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
            lines.append(f"  next   : add {missing[0]} "
                         f"(then {', '.join(missing[1:])})" if len(missing) > 1
                         else f"  next   : add {missing[0]}")
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
            lines.append(f"  {cube['label']:<12} conf={cube['conf']:.2f} "
                         f"centre=({cube['cx']:.0f},{cube['cy']:.0f}) "
                         f"box={cube['w']:.0f}x{cube['h']:.0f}  [{role}]")
        lines.append("")

    lines.append(f"guidance: {build_guidance(groups)}")

    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main():
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else script_dir / DEFAULT_IMAGE
    image_path = image_path.resolve()
    if not image_path.is_file():
        sys.exit(f"Image not found: {image_path}")

    output_folder = (script_dir / OUTPUT_FOLDER).resolve()
    output_folder.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        sys.exit(f"Could not read image: {image_path}")
    if RESIZE is not None:
        image = cv2.resize(image, RESIZE)

    model = YOLO(MODEL_PATH)
    cubes = detect_cubes(model, image)
    groups = group_cubes(cubes)

    report = build_report(image_path.name, groups)
    print(report)

    annotate(image, groups)
    out_img = output_folder / image_path.name
    out_txt = output_folder / f"{image_path.stem}.txt"
    cv2.imwrite(str(out_img), image)
    out_txt.write_text(report)

    print(f"Saved {out_img}")
    print(f"Saved {out_txt}")


if __name__ == '__main__':
    main()

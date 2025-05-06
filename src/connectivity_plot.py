#!/usr/bin/env python3
"""
connectivity_plot.py

Generate connectivity plots from PEB .mat files.
Usage:
    python connectivity_plot.py --mat-files file1.mat file2.mat --outdir figures
"""
import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.sparse import issparse
import networkx as nx
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# -------------------------------
# Adjustable parameters
# -------------------------------
ICON_ZOOM = 0.35                # size of the node icons
SHAPE_ICON_ZOOM = 0.1           # zoom for ion-channel icons
SHAPE_ARROW_LENGTH = 1          # length of ion-channel arrows
GRID_SPACING = 8                # distance between region centers
REGION_RADIUS = 3               # radius of each brain-area circle
BASE_OFFSET = 1.2               # perp-offset for extrinsic arrows
LABEL_PADDING = 0.7             # extra push for extrinsic labels (perp)
PARALLEL_PADDING = 0.3          # push for extrinsic labels (along slanted links)
VERTICAL_LABEL_PAD = 0.9        # push for labels on vertical links
SCALE_INTRINSIC = 2.1           # scaling of intrinsic node layout
INTRINSIC_ARROW_PADDING = 0.8   # trim off ends of intrinsic arrows
CIRCLE_HEAD_RADIUS = 0.18       # radius for self-loop arrow-heads
CIRCLE_POINTER_SIZE = CIRCLE_HEAD_RADIUS * 2
FIG_WIDTH_PER = 5               # inches per covariate column
FIG_HEIGHT = 6                  # inches total height

# Offsets for specific ion channels
ION_CHANNEL_Y_OFFSETS = {
    1: -0.12,
}

# Self-loop offsets (loop and label separately)
SELF_LOOP_OFFSETS = {
    1: (-0.4, 0.7),
    2: (0.7, 0.0),
    3: (0.0, 0.7),
    4: (0.8, 0.0),
}
SELF_LOOP_LABEL_OFFSETS = {
    1: (-0.4, 1.2),
    2: (1.0, 0.2),
    3: (0.0, 1.2),
    4: (1.0, 0.2),
}

# Mapping raw covariate keys to titles
RAW_TITLE_MAP = {
    'covariate1': 'baseline',
    'covariate2': 'sustained',
    'covariate3': 'transient',
}

# Layout definitions
REGION_POSITIONS = {
    1: (0, GRID_SPACING),
    2: (GRID_SPACING, GRID_SPACING),
    3: (0, 0),
    4: (GRID_SPACING, 0),
}
INTRINSIC_BASE_POSITIONS = {
    1: (-1, 0.5),
    2: (0, 2.0),
    3: (1, 1.0),
    4: (0, 0),
}
CELL_NODES = {1: "SS", 2: "SP", 3: "II", 4: "DP"}

# Default image paths (override via CLI)
DEFAULT_NODE_IMAGES = {}
DEFAULT_SHAPE_IMAGES = {}


def fmt_trunc_str(x: float, ndigits: int = 2) -> str:
    """
    Truncate float x to a string with ndigits decimals without rounding.
    """
    s = f"{x:.{ndigits+2}f}"  # extra precision for safe truncation
    whole, _, frac = s.partition('.')
    return f"{whole}.{(frac + '0'*ndigits)[:ndigits]}"


def draw_self_loop(ax, position, loop_radius=0.4, color='#008080',
                   lw=2, offset=(0, 0), angle_deg=270):
    """
    Draw a circular self-loop with an arrow-head.
    """
    cx, cy = position[0] + offset[0], position[1] + offset[1]
    circle = Circle((cx, cy), loop_radius,
                    edgecolor=color, facecolor='none', lw=lw, zorder=12)
    ax.add_patch(circle)
    theta = np.deg2rad(angle_deg)
    head_x = cx + loop_radius * np.cos(theta)
    head_y = cy + loop_radius * np.sin(theta)
    head = Circle((head_x, head_y), CIRCLE_POINTER_SIZE / 2,
                  edgecolor=color, facecolor=color, zorder=13)
    ax.add_patch(head)


def process_file(mat_file: str) -> list:
    """
    Load a .mat file and extract segments of extrinsic, intrinsic, and shape connectivity.
    """
    mat = sio.loadmat(mat_file, squeeze_me=True, struct_as_record=False)
    peb = mat['Results'].PEB_thresholded

    def to_float(v):
        return float(v.toarray()[0, 0]) if issparse(v) else float(v)

    def to_str(v):
        return ''.join(v.astype(str)).strip() if isinstance(v, np.ndarray) else str(v)

    Ep = np.array([to_float(x) for x in peb.Ep.flatten()])
    Pnames = [to_str(x) for x in peb.Pnames.flatten()]
    Np = len(Pnames)
    nseg = len(Ep) // Np if len(Ep) % Np == 0 else 1

    pat_ex = re.compile(r'(A|AN)\{(\d+)\}\((\d+),(\d+)\)')
    pat_shape = re.compile(r'Covariate (\d+):T\((\d+),(\d+)\)')

    segments = []
    for s in range(nseg):
        Ep_s = Ep[s*Np:(s+1)*Np]
        cov_ex, cov_int, cov_shape = {}, {}, {}

        for val, pn in zip(Ep_s, Pnames):
            if val != 0 and ':' in pn:
                cov, conn = pn.split(':', 1)
                m = pat_ex.search(conn)
                if m:
                    typ, ins, tgt, src = m.groups()
                    cov_ex.setdefault(cov, []).append((int(src), int(tgt), typ + ins, val))
            if val != 0 and pn.startswith('Covariate'):
                cov, conn = pn.split(':', 1)
                if conn.strip().startswith('H'):
                    cov_int.setdefault(cov, []).append({'label': conn.strip(), 'Ep': val})
            m2 = pat_shape.match(pn)
            if m2:
                cov_idx, area, shape_idx = map(int, m2.groups())
                key = f"Covariate {cov_idx}"
                cov_shape.setdefault(key, []).append((area, shape_idx, val))

        segments.append({
            'cov_extrinsic': cov_ex,
            'covariate_groups': cov_int,
            'cov_shape': cov_shape,
            'segment': s + 1,
        })

    return segments


def plot_covariate(ax, cov: str, data: dict, node_images: dict, shape_images: dict):
    """
    Plot connectivity (extrinsic, intrinsic, and shapes) for a single covariate.
    """
    ax.set_aspect('equal', 'box')
    ax.axis('off')

    # Draw brain regions
    for pos in REGION_POSITIONS.values():
        ax.add_patch(Circle(pos, REGION_RADIUS, facecolor='none', edgecolor='black', lw=0))

    # Ion-channel shapes and arrows
    for area, shapes in data.get('cov_shape', {}).get(cov, []):
        for shape_idx, ep_val in shapes:
            # Position and draw the shape icon
            # ... (omitted for brevity)
            pass

    # Extrinsic connections
    # ... (omitted for brevity)

    # Intrinsic connections and self-loops
    # ... (omitted for brevity)

    # Adjust plot limits
    xs = [p[0] for p in REGION_POSITIONS.values()]
    ys = [p[1] for p in REGION_POSITIONS.values()]
    pad = REGION_RADIUS + 1
    extra = SHAPE_ARROW_LENGTH + max(abs(v) for v in ION_CHANNEL_Y_OFFSETS.values())
    ax.set_xlim(min(xs) - pad - extra, max(xs) + pad + extra)
    ax.set_ylim(min(ys) - pad - extra, max(ys) + pad + extra)
    ax.margins(0.05)


def main():
    parser = argparse.ArgumentParser(
        description='Generate connectivity plots from PEB .mat files.'
    )
    parser.add_argument(
        '--mat-files', '-m', nargs='+', required=True,
        help='Paths to .mat files'
    )
    parser.add_argument(
        '--outdir', '-o', required=True,
        help='Directory to save figures'
    )
    parser.add_argument(
        '--node-images', nargs=4, metavar=('SS', 'SP', 'II', 'DP'),
        help='Paths to node icon images'
    )
    parser.add_argument(
        '--shape-images', nargs='+',
        help='Paths to shape icon images'
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    node_imgs = dict(enumerate(args.node_images or [], start=1))
    shape_imgs = dict(enumerate(args.shape_images or [], start=1))

    all_segments = [process_file(fp) for fp in args.mat_files]
    covariates = sorted({
        cov for segs in all_segments for seg in segs
        for cov in (*seg['cov_extrinsic'], *seg['covariate_groups'], *seg['cov_shape'])
    })

    for idx, segs in enumerate(all_segments, start=1):
        for seg in segs:
            fig, axes = plt.subplots(
                1, len(covariates),
                figsize=(FIG_WIDTH_PER * len(covariates), FIG_HEIGHT),
                squeeze=False
            )
            axes = axes.flatten()
            for ax in axes:
                ax.set_facecolor('none')

            for ax, cov in zip(axes, covariates):
                plot_covariate(ax, cov, seg, node_imgs or DEFAULT_NODE_IMAGES,
                               shape_imgs or DEFAULT_SHAPE_IMAGES)
                key = cov.replace(' ', '').lower()
                ax.set_title(RAW_TITLE_MAP.get(key, cov), fontsize=10)

            fig.suptitle(f"File {idx} – Segment {seg['segment']}", fontsize=14)
            fig.subplots_adjust(left=0.05, right=0.98, top=0.9, bottom=0.05, wspace=0)
            fname = f"file{idx}_segment{seg['segment']}.svg"
            outpath = os.path.join(args.outdir, fname)
            fig.savefig(outpath, format='svg', bbox_inches='tight', transparent=True)
            print(f"Saved: {outpath}")

if __name__ == '__main__':
    main()

"""
diagnose_junctions.py

Zooms into missed junctions and shows exactly what the skeleton looks like there.
Also tests different pre-dilation amounts to find the minimum that recovers all branches.

Usage:
    python diagnose_junctions.py --tif test.tif --out /tmp/junc
"""
import argparse, os
import numpy as np
from PIL import Image
from skimage import morphology
from scipy.ndimage import convolve
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def load_tif_2d(path):
    try:
        import tifffile
        arr = tifffile.imread(path)
    except ImportError:
        arr = np.array(Image.open(path).convert('L'))
    if arr.ndim == 3:
        if arr.shape[0] <= 4:   arr = arr[0]
        elif arr.shape[2] <= 4: arr = (0.299*arr[:,:,0]+0.587*arr[:,:,1]+0.114*arr[:,:,2]).astype(arr.dtype)
        else:                   arr = arr.max(axis=0)
    return arr


def skeleton_stats(binary, dilation_r=0):
    """Skeletonize with optional pre-dilation, return skel + branch coords."""
    mask = binary > 0
    if dilation_r > 0:
        mask = morphology.dilation(mask, morphology.disk(dilation_r))
    skel = morphology.skeletonize(mask)
    kernel = np.ones((3,3), dtype=np.uint8); kernel[1,1] = 0
    nc = convolve(skel.astype(np.uint8), kernel, mode='constant', cval=0)
    branch_mask = skel & (nc >= 3)
    by, bx = np.where(branch_mask)
    return skel, branch_mask, nc, np.column_stack((bx.astype(float), by.astype(float))) if len(by) else np.zeros((0,2))


def cluster(coords, r=4.0):
    if len(coords) == 0: return coords
    tree = cKDTree(coords)
    visited = np.zeros(len(coords), dtype=bool)
    out = []
    for i in range(len(coords)):
        if visited[i]: continue
        nb = tree.query_ball_point(coords[i], r)
        out.append(coords[nb].mean(axis=0))
        visited[nb] = True
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tif',  required=True)
    ap.add_argument('--out',  default='/tmp/junc')
    ap.add_argument('--threshold', type=int, default=128)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    raw  = load_tif_2d(args.tif)
    binary = (raw > args.threshold).astype(np.uint8)
    H, W = raw.shape
    print(f"Image: {H}x{W}  membrane px: {np.sum(binary)}")

    # ── Figure 1: effect of dilation on branch count ─────────────────────
    dils = [0, 1, 2, 3]
    fig, axes = plt.subplots(1, len(dils), figsize=(5*len(dils), 5))
    for ax, dr in zip(axes, dils):
        skel, branch_mask, nc, bcoords = skeleton_stats(binary, dr)
        verts = cluster(bcoords, r=4.0)
        disp = np.zeros((H, W, 3), dtype=float)
        disp[skel & (nc==1)] = [0.2, 0.2, 1.0]   # endpoint blue
        disp[skel & (nc==2)] = [0.0, 0.8, 0.0]   # edge green
        disp[skel & (nc>=3)] = [1.0, 0.0, 0.0]   # branch red
        ax.imshow(disp)
        if len(verts):
            ax.scatter(verts[:,0], verts[:,1], c='yellow', s=12,
                       zorder=5, linewidths=0.3, edgecolors='black')
        ax.set_title(f'dilation=disk({dr})\nbranch px={np.sum(branch_mask)}'
                     f'  V={len(verts)}', fontsize=10)
        ax.axis('off')
        print(f"disk({dr}): branch_px={np.sum(branch_mask):5d}  vertices={len(verts):4d}  skel_px={np.sum(skel):6d}")
    plt.suptitle('Effect of pre-dilation on branch point detection\n'
                 '(red=branch, green=edge, blue=endpoint, yellow=clustered vertex)', fontsize=11)
    plt.tight_layout()
    p = os.path.join(args.out, 'dilation_comparison.png')
    plt.savefig(p, dpi=130, bbox_inches='tight'); plt.close()
    print(f"Saved: {p}")

    # ── Figure 2: zoom into missed junctions (low-branch regions) ────────
    # Find areas with dense endpoints but no nearby branch point
    skel0, bm0, nc0, bc0 = skeleton_stats(binary, 0)
    skel1, bm1, nc1, bc1 = skeleton_stats(binary, 1)

    # endpoint pixels in dilation=0 skeleton
    ep_mask = skel0 & (nc0 == 1)
    ep_y, ep_x = np.where(ep_mask)

    # Branch pixels in dilation=1 skeleton (the "truth")
    br1_y, br1_x = np.where(bm1)

    if len(ep_y) > 0:
        # Find endpoint clusters — each cluster is a potential missed junction
        ep_coords = np.column_stack((ep_x.astype(float), ep_y.astype(float)))
        ep_clustered = cluster(ep_coords, r=20.0)  # group nearby endpoints
        # Score: how many endpoints are in this cluster
        tree = cKDTree(ep_coords)
        scores = [len(tree.query_ball_point(c, 20.0)) for c in ep_clustered]
        top_idx = np.argsort(scores)[::-1][:9]
        top_junctions = ep_clustered[top_idx]

        print(f"\nTop missed junction candidates (by nearby endpoint count):")
        for i, (cx, cy) in enumerate(top_junctions):
            print(f"  [{i}] center=({cx:.0f},{cy:.0f})  nearby_endpoints={scores[top_idx[i]]}")

        pad = 40
        n_show = min(9, len(top_junctions))
        nrows = (n_show + 2) // 3
        fig, axes = plt.subplots(nrows, 3, figsize=(12, 4*nrows))
        axes = np.array(axes).flatten()

        for i in range(n_show):
            cx, cy = int(top_junctions[i][0]), int(top_junctions[i][1])
            y0, y1 = max(0, cy-pad), min(H, cy+pad)
            x0, x1 = max(0, cx-pad), min(W, cx+pad)

            ax = axes[i]
            # Background: grayscale crop
            ax.imshow(raw[y0:y1, x0:x1], cmap='gray', alpha=0.6)

            # Overlay dilation=0 skeleton (green)
            crop0 = skel0[y0:y1, x0:x1]
            ys0, xs0 = np.where(crop0); ax.scatter(xs0, ys0, c='lime', s=2, alpha=0.7)

            # Overlay dilation=1 skeleton (magenta) for comparison
            crop1 = skel1[y0:y1, x0:x1]
            ys1, xs1 = np.where(crop1); ax.scatter(xs1, ys1, c='magenta', s=2, alpha=0.5)

            # Branch pixels (dilation=0): red; dilation=1: yellow
            br0c = bm0[y0:y1, x0:x1]; yb0,xb0 = np.where(br0c)
            br1c = bm1[y0:y1, x0:x1]; yb1,xb1 = np.where(br1c)
            ax.scatter(xb0, yb0, c='red',    s=20, zorder=10)
            ax.scatter(xb1, yb1, c='yellow', s=20, zorder=10, marker='*')

            ax.set_title(f'Junction {i}  ({cx},{cy})\n'
                         f'red=branch(d=0)  yellow★=branch(d=1)', fontsize=8)
            ax.set_xlim(0, x1-x0); ax.set_ylim(y1-y0, 0)
            ax.axis('off')

        for j in range(n_show, len(axes)):
            axes[j].axis('off')

        patches = [mpatches.Patch(color='lime',    label='skel dilation=0'),
                   mpatches.Patch(color='magenta', label='skel dilation=1'),
                   mpatches.Patch(color='red',     label='branch d=0'),
                   mpatches.Patch(color='yellow',  label='branch d=1')]
        fig.legend(handles=patches, loc='lower right', fontsize=9)
        plt.suptitle('Missed junctions: dilation=0 has endpoint but no branch\n'
                     'yellow stars = branches recovered by disk(1) dilation', fontsize=11)
        plt.tight_layout()
        p2 = os.path.join(args.out, 'missed_junctions_zoom.png')
        plt.savefig(p2, dpi=130, bbox_inches='tight'); plt.close()
        print(f"Saved: {p2}")

    # ── Print recommendation ──────────────────────────────────────────────
    counts = []
    for dr in dils:
        _, _, _, bc = skeleton_stats(binary, dr)
        counts.append(len(cluster(bc, r=4.0)))
    print(f"\nVertex counts by dilation: {dict(zip(dils, counts))}")
    best = dils[int(np.argmax(counts))]
    print(f"→ Recommended dilation: disk({best})  (maximizes vertex count)")
    print(f"\nIn topology.py _labels_to_boundary or extract_topology, add:")
    print(f"    boundary = morphology.dilation(boundary, morphology.disk({best}))")


if __name__ == '__main__':
    main()

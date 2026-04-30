"""
Time-series force inference.

Each frame is solved independently (normalized tensions, mean ≈ 1).
Cross-frame *scale alignment* recovers relative absolute tensions so that
evolution can be tracked.

Three alignment strategies are provided:

``"shared_edges"``  (default, no extra data)
    Cells that persist between consecutive frames share edges.  The
    per-frame multiplicative scale factor is chosen to minimise the
    log-ratio of tensions on shared edges (least-squares on shared pairs).

``"fluorescence"``
    Pass a co-registered fluorescence image (e.g. myosin-II channel).
    The mean pixel intensity along each edge is used as a proxy for
    absolute tension.  A single linear regression ``T = α · I + β``
    calibrated on one reference frame sets the global scale.

``"fixed_edge"``
    The user nominates a specific cell-pair (or a list of pairs) whose
    tension is assumed constant (e.g. a quiescent reference junction).
    All frames are scaled so that pair's tension equals its value in
    frame 0.

Usage example
-------------
>>> from force_inference.timeseries import TimeSeries, align_timeseries
>>> ts = TimeSeries()
>>> for frame_tissue, frame_result in zip(tissues, results):
...     ts.add_frame(frame_tissue, frame_result)
>>> ts.align(strategy="shared_edges")
>>> trajectories = ts.edge_trajectories()  # dict (c1,c2) -> array of T per frame
>>> ts.plot_trajectories(ax, top_n=10)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from .core import Tissue, ForceResult

logger = logging.getLogger("ForceInference.TimeSeries")


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class TimeSeries:
    """Collection of (Tissue, ForceResult) frames with optional timestamps."""

    tissues: List[Tissue]       = field(default_factory=list)
    results: List[ForceResult]  = field(default_factory=list)
    times:   List[float]        = field(default_factory=list)   # physical time (e.g. minutes)

    # Per-frame multiplicative scale factors set by align().
    # scale[i] converts frame i's normalised tensions to the reference scale:
    #   T_aligned[i] = scale[i] * result[i].tensions
    scales: Optional[np.ndarray] = None

    def add_frame(
        self,
        tissue: Tissue,
        result: ForceResult,
        time: float = None,
    ) -> None:
        """Append one solved frame."""
        self.tissues.append(tissue)
        self.results.append(result)
        self.times.append(float(time if time is not None else len(self.times)))

    def __len__(self) -> int:
        return len(self.tissues)

    # ------------------------------------------------------------------
    # Scale alignment
    # ------------------------------------------------------------------

    def align(
        self,
        strategy: str = "shared_edges",
        reference_frame: int = 0,
        fluorescence_images: Optional[List[np.ndarray]] = None,
        fixed_pairs: Optional[List[Tuple[int, int]]] = None,
    ) -> "TimeSeries":
        """
        Compute per-frame scale factors so that tensions are comparable across
        frames.  Sets ``self.scales`` in-place and returns self.

        Parameters
        ----------
        strategy : {"shared_edges", "fluorescence", "fixed_edge", "median"}
        reference_frame : int
            Index of the frame that defines the reference scale (default 0).
        fluorescence_images : list of (H,W) arrays
            Required when strategy == "fluorescence".
        fixed_pairs : list of (cell_label_1, cell_label_2)
            Required when strategy == "fixed_edge".
        """
        n = len(self)
        if n == 0:
            return self

        if strategy == "shared_edges":
            self.scales = _align_shared_edges(self, reference_frame)
        elif strategy == "fluorescence":
            if fluorescence_images is None:
                raise ValueError("fluorescence_images required for strategy='fluorescence'")
            self.scales = _align_fluorescence(self, fluorescence_images, reference_frame)
        elif strategy == "fixed_edge":
            if fixed_pairs is None:
                raise ValueError("fixed_pairs required for strategy='fixed_edge'")
            self.scales = _align_fixed_pairs(self, fixed_pairs, reference_frame)
        elif strategy == "median":
            self.scales = _align_median(self, reference_frame)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        logger.info("Scale alignment (%s): %s", strategy,
                    ", ".join(f"f{i}={s:.3f}" for i, s in enumerate(self.scales)))
        return self

    # ------------------------------------------------------------------
    # Tension retrieval
    # ------------------------------------------------------------------

    def aligned_tensions(self, frame: int) -> np.ndarray:
        """
        Return tensions for *frame* after applying the scale factor.
        If align() has not been called, returns the raw normalised tensions.
        """
        T = self.results[frame].tensions.copy()
        if self.scales is not None:
            T = T * self.scales[frame]
        return T

    def edge_trajectories(
        self,
        min_frames: int = 2,
    ) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Build a dictionary mapping each cell-pair (c1, c2) to a float array
        of length ``len(self)`` containing the aligned tension at each frame
        (NaN when the edge is absent or has NaN tension in that frame).

        Parameters
        ----------
        min_frames : int
            Exclude trajectories that have valid data in fewer than this many
            frames (filters out transient or border edges).
        """
        if len(self) == 0:
            return {}

        # Collect all cell pairs seen across all frames
        all_pairs: set = set()
        for tissue in self.tissues:
            for c1, c2 in tissue.E_cells:
                if c1 > 0 and c2 > 0:
                    all_pairs.add((min(int(c1), int(c2)), max(int(c1), int(c2))))

        trajectories: Dict[Tuple[int, int], np.ndarray] = {}

        for pair in all_pairs:
            traj = np.full(len(self), np.nan)
            for fi, (tissue, result) in enumerate(zip(self.tissues, self.results)):
                T = self.aligned_tensions(fi)
                t_val = _get_pair_tension(tissue, T, pair)
                traj[fi] = t_val

            n_valid = int(np.sum(~np.isnan(traj)))
            if n_valid >= min_frames:
                trajectories[pair] = traj

        return trajectories

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot_trajectories(
        self,
        ax: plt.Axes,
        top_n: int = 20,
        highlight_pairs: Optional[List[Tuple[int, int]]] = None,
        cmap: str = "turbo",
        alpha: float = 0.55,
        lw: float = 1.0,
        show_variance: bool = True,
    ) -> plt.Axes:
        """
        Plot per-edge tension trajectories on *ax*.

        ``top_n`` edges with the highest temporal variance are drawn coloured;
        the rest are shown as faint gray lines.  Specific pairs can be
        highlighted with ``highlight_pairs``.

        Parameters
        ----------
        show_variance : bool
            If True, shade the ±1σ band of all trajectories in the background.
        """
        trajs = self.edge_trajectories()
        if not trajs:
            logger.warning("No trajectories to plot.")
            return ax

        t_arr = np.array(self.times)
        all_T = np.array([v for v in trajs.values()])  # (n_pairs, n_frames)

        # Background variance band
        if show_variance and all_T.shape[0] > 1:
            mean_T = np.nanmean(all_T, axis=0)
            std_T  = np.nanstd(all_T, axis=0)
            ax.fill_between(t_arr, mean_T - std_T, mean_T + std_T,
                            color="gray", alpha=0.15, zorder=0, label="±1σ band")
            ax.plot(t_arr, mean_T, color="gray", lw=1.5, ls="--",
                    alpha=0.6, zorder=1, label="mean")

        # Rank by temporal variance (most dynamic edges on top)
        variances = {pair: float(np.nanvar(traj))
                     for pair, traj in trajs.items()}
        sorted_pairs = sorted(variances, key=variances.get, reverse=True)

        cmap_fn  = get_cmap(cmap)
        top_set  = set(sorted_pairs[:top_n])
        hi_set   = set(map(tuple, highlight_pairs)) if highlight_pairs else set()

        # Gray background lines
        for pair, traj in trajs.items():
            if pair not in top_set and pair not in hi_set:
                ax.plot(t_arr, traj, color="lightgray", lw=0.5, alpha=0.3, zorder=1)

        # Coloured top-N lines
        for rank, pair in enumerate(sorted_pairs[:top_n]):
            traj = trajs[pair]
            colour = cmap_fn(rank / max(top_n - 1, 1))
            lbl = f"({pair[0]},{pair[1]})" if pair in hi_set else None
            ax.plot(t_arr, traj, color=colour, lw=lw, alpha=alpha,
                    zorder=2, label=lbl)

        # Highlighted pairs (drawn on top, thicker)
        for pair in hi_set:
            if pair in trajs:
                ax.plot(t_arr, trajs[pair], color="white", lw=lw * 3,
                        alpha=0.9, zorder=3)
                ax.plot(t_arr, trajs[pair], color="red", lw=lw * 2,
                        alpha=1.0, zorder=4, label=f"{pair}")

        ax.set_xlabel("Time")
        ax.set_ylabel("Tension (aligned)")
        ax.set_title(f"Tension trajectories  ·  top {top_n} by variance shown")
        if hi_set:
            ax.legend(fontsize=8)
        return ax

    def plot_scale_factors(self, ax: plt.Axes) -> plt.Axes:
        """Bar chart of per-frame scale factors (useful sanity-check)."""
        if self.scales is None:
            ax.text(0.5, 0.5, "align() not called yet",
                    transform=ax.transAxes, ha="center", va="center")
            return ax
        t_arr = np.array(self.times)
        ax.bar(t_arr, self.scales, width=np.diff(t_arr).min() * 0.7
               if len(t_arr) > 1 else 0.8,
               color="steelblue", alpha=0.8)
        ax.axhline(1.0, color="gray", ls="--", lw=1)
        ax.set_xlabel("Time")
        ax.set_ylabel("Scale factor")
        ax.set_title("Per-frame scale factors relative to reference frame")
        return ax


# ---------------------------------------------------------------------------
# Alignment strategies (internal)
# ---------------------------------------------------------------------------

def _edge_pair_map(tissue: Tissue) -> Dict[Tuple[int, int], List[int]]:
    """Return {(min_c, max_c): [edge_indices]} for all non-border edges."""
    m: Dict[Tuple[int, int], List[int]] = {}
    for i, (c1, c2) in enumerate(tissue.E_cells):
        if c1 > 0 and c2 > 0:
            key = (min(int(c1), int(c2)), max(int(c1), int(c2)))
            m.setdefault(key, []).append(i)
    return m


def _get_pair_tension(
    tissue: Tissue,
    tensions: np.ndarray,
    pair: Tuple[int, int],
) -> float:
    """Mean non-NaN tension for a given cell pair in one frame."""
    c_lo, c_hi = pair
    vals = []
    for i, (c1, c2) in enumerate(tissue.E_cells):
        key = (min(int(c1), int(c2)), max(int(c1), int(c2)))
        if key == (c_lo, c_hi):
            t = float(tensions[i])
            if np.isfinite(t):
                vals.append(t)
    return float(np.mean(vals)) if vals else np.nan


def _align_shared_edges(ts: TimeSeries, ref: int) -> np.ndarray:
    """
    Per-frame scale factors from shared-cell-pair log-ratio minimisation.

    For each consecutive pair of frames (f, f+1) we find all cell pairs
    present in both.  We solve:

        min_s  Σ_{shared pairs} (log T_{f+1,e} + log s - log T_{f,e})²

    giving  log s = mean(log T_{f,e} - log T_{f+1,e})  over shared pairs,
    i.e.  s = exp(mean log-ratio).

    Scale factors are chained from the reference frame outward.
    """
    n = len(ts)
    log_scales = np.zeros(n)  # log(scale[i] / scale[ref])

    # Build pair → tension maps for every frame
    pair_T: List[Dict[Tuple[int, int], float]] = []
    for fi in range(n):
        T = ts.results[fi].tensions
        m = _edge_pair_map(ts.tissues[fi])
        pt: Dict[Tuple[int, int], float] = {}
        for pair, idxs in m.items():
            vals = [T[i] for i in idxs if np.isfinite(T[i]) and T[i] > 0]
            if vals:
                pt[pair] = float(np.mean(vals))
        pair_T.append(pt)

    # Forward pass: chain log-ratios from frame 0 to n-1
    for fi in range(n - 1):
        shared = set(pair_T[fi]) & set(pair_T[fi + 1])
        if not shared:
            logger.warning("Frames %d and %d share no edges — scale unchanged.", fi, fi + 1)
            log_scales[fi + 1] = log_scales[fi]
            continue
        log_ratios = []
        for pair in shared:
            t_cur  = pair_T[fi][pair]
            t_next = pair_T[fi + 1][pair]
            if t_cur > 0 and t_next > 0:
                log_ratios.append(np.log(t_cur) - np.log(t_next))
        if log_ratios:
            log_scales[fi + 1] = log_scales[fi] + float(np.median(log_ratios))
        else:
            log_scales[fi + 1] = log_scales[fi]

    # Normalise so that reference frame has scale = 1
    log_scales -= log_scales[ref]
    scales = np.exp(log_scales)
    return scales


def _align_median(ts: TimeSeries, ref: int) -> np.ndarray:
    """
    Simplest normalisation: scale each frame so its median (non-NaN) tension
    equals the reference frame's median.  Assumes the tissue-average tension
    is conserved — only valid for gentle, slowly-evolving perturbations.
    """
    n = len(ts)
    ref_median = float(np.nanmedian(ts.results[ref].tensions))
    if ref_median <= 0:
        logger.warning("Reference frame median tension ≤ 0; using scale=1.")
        return np.ones(n)
    scales = np.ones(n)
    for fi in range(n):
        m = float(np.nanmedian(ts.results[fi].tensions))
        if m > 0:
            scales[fi] = ref_median / m
    return scales


def _align_fluorescence(
    ts: TimeSeries,
    fluor_imgs: List[np.ndarray],
    ref: int,
) -> np.ndarray:
    """
    Use mean fluorescence intensity along each edge as a proxy for tension.
    Fit T = α·I on the reference frame (through the origin, zero-intercept)
    then for every other frame compute the scale factor that minimises
    Σ (s·T_i - α·I_i)².

    Requires co-registered fluorescence images (same pixel grid as labels).
    """
    from scipy.ndimage import map_coordinates

    n = len(ts)

    def _edge_intensities(tissue: Tissue, img: np.ndarray) -> np.ndarray:
        """Mean fluorescence intensity per edge (NaN for edges with no pixels)."""
        vals = np.full(len(tissue.E), np.nan)
        for i, (v1, v2) in enumerate(tissue.E):
            if tissue.E_pixels is not None and len(tissue.E_pixels[i]) > 1:
                px = np.asarray(tissue.E_pixels[i])
                coords = [px[:, 1].clip(0, img.shape[0] - 1),
                          px[:, 0].clip(0, img.shape[1] - 1)]
                intensities = map_coordinates(img.astype(float), coords, order=1)
                vals[i] = float(np.mean(intensities))
        return vals

    # Reference frame: fit α via least squares T = α · I  (no intercept)
    I_ref = _edge_intensities(ts.tissues[ref], fluor_imgs[ref])
    T_ref = ts.results[ref].tensions
    valid = np.isfinite(I_ref) & np.isfinite(T_ref) & (I_ref > 0) & (T_ref > 0)
    if not np.any(valid):
        logger.warning("No valid fluorescence/tension pairs in reference frame; using scale=1.")
        return np.ones(n)

    alpha = float(np.dot(I_ref[valid], T_ref[valid]) / np.dot(I_ref[valid], I_ref[valid]))
    logger.info("Fluorescence calibration: α = %.4f  (T = α·I)", alpha)

    scales = np.ones(n)
    for fi in range(n):
        I_fi = _edge_intensities(ts.tissues[fi], fluor_imgs[fi])
        T_fi = ts.results[fi].tensions
        v = np.isfinite(I_fi) & np.isfinite(T_fi) & (I_fi > 0) & (T_fi > 0)
        if not np.any(v):
            continue
        # Best scale s minimises Σ (s·T - α·I)²  → s = Σ(α·I·T) / Σ(T²)
        s = float(np.dot(alpha * I_fi[v], T_fi[v]) / np.dot(T_fi[v], T_fi[v]))
        scales[fi] = s

    # Normalise to reference
    scales /= scales[ref]
    return scales


def _align_fixed_pairs(
    ts: TimeSeries,
    fixed_pairs: List[Tuple[int, int]],
    ref: int,
) -> np.ndarray:
    """
    Scale each frame so that the mean tension of the nominated cell pairs
    equals their value in the reference frame.  Suitable when laser ablation
    or some other method has identified edges with known/stable tension.
    """
    n = len(ts)
    fixed_pairs = [(min(a, b), max(a, b)) for a, b in fixed_pairs]

    def _anchor_tension(fi: int) -> float:
        T = ts.results[fi].tensions
        vals = []
        for pair in fixed_pairs:
            v = _get_pair_tension(ts.tissues[fi], T, pair)
            if np.isfinite(v) and v > 0:
                vals.append(v)
        return float(np.mean(vals)) if vals else np.nan

    anchor_ref = _anchor_tension(ref)
    if not np.isfinite(anchor_ref) or anchor_ref <= 0:
        logger.warning("Fixed-pair tension in reference frame is invalid; using scale=1.")
        return np.ones(n)

    scales = np.ones(n)
    for fi in range(n):
        a = _anchor_tension(fi)
        if np.isfinite(a) and a > 0:
            scales[fi] = anchor_ref / a
    return scales


# ---------------------------------------------------------------------------
# Convenience function — full pipeline on a list of label arrays
# ---------------------------------------------------------------------------

def align_timeseries(
    tissues: List[Tissue],
    results: List[ForceResult],
    times: Optional[List[float]] = None,
    strategy: str = "shared_edges",
    **align_kwargs,
) -> TimeSeries:
    """
    Build and align a :class:`TimeSeries` from lists of tissues and results.

    Parameters
    ----------
    tissues, results : lists of equal length
    times : optional list of physical timestamps (e.g. minutes)
    strategy : passed to :meth:`TimeSeries.align`
    **align_kwargs : forwarded to :meth:`TimeSeries.align`

    Returns
    -------
    TimeSeries with ``scales`` set.
    """
    ts = TimeSeries()
    for fi, (tissue, result) in enumerate(zip(tissues, results)):
        t = times[fi] if times is not None else float(fi)
        ts.add_frame(tissue, result, time=t)
    ts.align(strategy=strategy, **align_kwargs)
    return ts

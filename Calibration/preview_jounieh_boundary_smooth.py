"""Preview smoothing options for Jounieh_Road_Boundaries.csv.

Does not overwrite the source file. Writes comparison figures to
Calibration/paper_summary/boundary_smooth/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "Lebanon_Jounieh" / "Jounieh_Road_Boundaries.csv"
OUT = ROOT / "Calibration" / "paper_summary" / "boundary_smooth"
INK = "#1a1a1a"
ORIG = "#B8C0C8"
SMOOTH = "#1F4E79"


def load_rings(path: Path) -> list[tuple[str, int, np.ndarray]]:
    df = pd.read_csv(path)
    rings = []
    for (kind, pid), g in df.groupby(["kind", "polygon_id"], sort=True):
        xy = g.sort_values("vertex_index")[["x", "y"]].to_numpy(float)
        if len(xy) and not np.allclose(xy[0], xy[-1]):
            xy = np.vstack([xy, xy[0]])
        rings.append((str(kind), int(pid), xy))
    return rings


def close_ring(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return xy
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    return xy


def open_ring(xy: np.ndarray) -> np.ndarray:
    xy = close_ring(xy)
    return xy[:-1] if len(xy) > 1 else xy


def chaikin(xy: np.ndarray, iters: int) -> np.ndarray:
    pts = open_ring(xy)
    for _ in range(iters):
        nxt = np.roll(pts, -1, axis=0)
        a = 0.75 * pts + 0.25 * nxt
        b = 0.25 * pts + 0.75 * nxt
        pts = np.empty((len(pts) * 2, 2), float)
        pts[0::2] = a
        pts[1::2] = b
    return close_ring(pts)


def spline(xy: np.ndarray, s: float, n: int = 400) -> np.ndarray:
    pts = open_ring(xy)
    if len(pts) < 4:
        return close_ring(pts)
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, per=True, quiet=True)
    u = np.linspace(0.0, 1.0, n, endpoint=False)
    xs, ys = splev(u, tck)
    return close_ring(np.column_stack([xs, ys]))


def gaussian(xy: np.ndarray, sigma_m: float, n: int = 400) -> np.ndarray:
    pts = open_ring(xy)
    if len(pts) < 3:
        return close_ring(pts)
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    peri = float(s[-1])
    if peri < 1e-6:
        return close_ring(pts)
    su = np.linspace(0.0, peri, n, endpoint=False)
    xs = np.interp(su, s, closed[:, 0])
    ys = np.interp(su, s, closed[:, 1])
    ds = peri / n
    sigma = max(sigma_m / ds, 0.4)
    pad = int(np.ceil(4 * sigma)) + 1
    xs = gaussian_filter1d(np.r_[xs[-pad:], xs, xs[:pad]], sigma)[pad:-pad]
    ys = gaussian_filter1d(np.r_[ys[-pad:], ys, ys[:pad]], sigma)[pad:-pad]
    return close_ring(np.column_stack([xs, ys]))


def round_buffer(xy: np.ndarray, radius: float) -> np.ndarray:
    poly = Polygon(open_ring(xy))
    if not poly.is_valid:
        poly = poly.buffer(0)
    out = poly.buffer(radius, join_style=1, resolution=16).buffer(-radius, join_style=1, resolution=16)
    if out.is_empty:
        return close_ring(xy)
    if out.geom_type == "MultiPolygon":
        out = max(out.geoms, key=lambda g: g.area)
    coords = np.asarray(out.exterior.coords, float)
    return close_ring(coords)


METHODS = (
    ("A original", lambda xy: close_ring(xy)),
    ("B Chaikin x2", lambda xy: chaikin(xy, 2)),
    ("C Chaikin x4", lambda xy: chaikin(xy, 4)),
    ("D spline light", lambda xy: spline(xy, s=1.5)),
    ("E spline medium", lambda xy: spline(xy, s=8.0)),
    ("F spline strong", lambda xy: spline(xy, s=28.0)),
    ("G round-buffer 0.35 m", lambda xy: round_buffer(xy, 0.35)),
    ("H round-buffer 0.70 m", lambda xy: round_buffer(xy, 0.70)),
    ("I Gaussian 0.8 m", lambda xy: gaussian(xy, 0.8)),
)


def _draw_rings(ax, orig, smooth, *, orig_lw=0.9, sm_lw=1.6):
    for _, _, xy in orig:
        ax.plot(xy[:, 0], xy[:, 1], color=ORIG, lw=orig_lw, zorder=1)
    for _, _, xy in smooth:
        ax.plot(xy[:, 0], xy[:, 1], color=SMOOTH, lw=sm_lw, zorder=2)


def plot() -> list[Path]:
    orig = load_rings(SRC)
    OUT.mkdir(parents=True, exist_ok=True)
    variants = []
    for name, fn in METHODS:
        rings = [(kind, pid, fn(xy)) for kind, pid, xy in orig]
        variants.append((name, rings))

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8.5})
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 10.4), constrained_layout=True)
    for ax, (name, rings) in zip(axes.ravel(), variants):
        _draw_rings(ax, orig, rings)
        ax.set_aspect("equal")
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#d5dbe3")
    grid = OUT / "jounieh_boundary_smooth_grid.png"
    fig.savefig(grid, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    zooms = (
        ("outer SW arm", (12.5, 28.5), (0.5, 12.5)),
        ("circulating island", (24.5, 36.5), (10.5, 22.8)),
        ("splitter island", (34.5, 43.0), (16.8, 22.6)),
        ("NE approach", (46.5, 56.2), (19.5, 32.0)),
    )
    fig, axes = plt.subplots(len(METHODS), len(zooms), figsize=(12.5, 18.5), constrained_layout=True)
    for r, (name, rings) in enumerate(variants):
        for c, (zname, xlim, ylim) in enumerate(zooms):
            ax = axes[r, c]
            _draw_rings(ax, orig, rings, orig_lw=0.8, sm_lw=1.7)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d5dbe3")
            if r == 0:
                ax.set_title(zname, fontsize=9)
            if c == 0:
                ax.set_ylabel(name, fontsize=8, color=INK)
    zoom = OUT / "jounieh_boundary_smooth_zooms.png"
    fig.savefig(zoom, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
    for _, _, xy in orig:
        ax.plot(xy[:, 0], xy[:, 1], color=SMOOTH, lw=1.5, zorder=2)
    ax.set_aspect("equal")
    ax.set_title("A original")
    ax.set_xticks([])
    ax.set_yticks([])
    original = OUT / "jounieh_boundary_original.png"
    fig.savefig(original, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    singles = [original]
    for name, rings in variants[1:]:
        tag = name.split(" ", 1)[0]
        fig, ax = plt.subplots(figsize=(8.4, 5.2), constrained_layout=True)
        _draw_rings(ax, orig, rings)
        ax.set_aspect("equal")
        ax.set_title(name)
        ax.set_xticks([])
        ax.set_yticks([])
        p = OUT / f"jounieh_boundary_{tag}.png"
        fig.savefig(p, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        singles.append(p)
    return [grid, zoom, *singles]


if __name__ == "__main__":
    for p in plot():
        print(p)

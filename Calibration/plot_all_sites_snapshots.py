"""Merged recorded-ego snapshots: three sites × t = 0, 3, 6 s.

    python Calibration/plot_all_sites_snapshots.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "Ref Images")]

from RL.corridor import oriented_box_corners  # noqa: E402

OUT_NAME = "all_sites_snapshots"
TRAIL_S = 3.0
TIMES_S = (0.0, 3.0, 6.0)
EGO = "#C0392B"
OTHER = "#A0A0A0"
BOUND = "#9AA5B1"
LANE = "#8AA0B8"
INK = "#1a1a1a"
TGSIM_SKIP_CLASS = {0.0, 1.0, 2.0}
TGSIM_CAR_CLASS = {3.0, 4.0, 5.0}
FT = 0.3048
CAR_LENGTH_M = (14.0 * FT, 17.0 * FT)
CAR_WIDTH_M = (5.5 * FT, 6.5 * FT)

SHOWCASE = (
    {
        "site": "freeway",
        "label": "Highway",
        "csv": ROOT / "data/Lebanon_Highway/Final_Lebanon_Data.csv",
        "nll": ROOT / "Calibration/diagnostics/per_id/per_id_best90_by_nll.csv",
        "ego_id": 9084,
        "default_size": (4.5, 1.8),
        "view_half": 52.0,
        "scale_m": 50,
        "t0_from_start": None,
    },
    {
        "site": "jounieh",
        "label": "Jounieh",
        "csv": ROOT / "data/Lebanon_Jounieh/prepared/trajectories_calibration.csv",
        "nll": ROOT / "Calibration/diagnostics_jounieh/per_id/per_id_best90_by_nll.csv",
        "ego_id": 12549,
        "default_size": (4.5, 1.8),
        "view_half": 45.0,
        "scale_m": 15,
        "focus_xy": (90.54, 49.68),
    },
    {
        "site": "tgsim",
        "label": "TGSIM",
        "csv": ROOT / "data/TGSIM FB/prepared/trajectories_calibration.csv",
        "nll": ROOT / "Calibration/diagnostics_tgsim/per_id/per_id_best90_by_nll.csv",
        "ego_id": 2980,
        "default_size": (6.2, 2.2),
        "view_half": 30.0,
        "scale_m": 20,
        "focus_xy": (38.0, 190.0),
        "t0_abs": 3210.0,
        "pin_focus": True,
    },
)
OUTPUTS = (
    ROOT / "Calibration" / "paper_summary",
    ROOT / "Paper Draft" / "ICLR" / "Appendix" / "calibration",
    ROOT / "Paper Draft" / "Appendix" / "calibration",
)
CANDIDATE_DIR = ROOT / "Calibration" / "paper_summary" / "snapshot_candidates"
# Freeway ids are passenger-car class 2 (the earlier showcase ids were class-1 trucks).
VARIANTS = tuple(
    {"tag": f"{i:02d}", "ids": {"freeway": fw, "jounieh": jo, "tgsim": tg}}
    for i, (fw, jo, tg) in enumerate(
        (
            (7041, 12549, 1745),
            (11486, 6739, 2570),
            (15547, 71171, 4061),
            (9155, 75317, 3131),
            (1421, 38595, 2641),
            (10559, 26889, 1353),
            (5885, 17252, 2817),
            (9084, 11076, 506),
            (69, 54381, 2105),
            (1566, 64373, 1260),
            (1601, 16882, 363),
            (273, 20503, 4402),
            (1283, 53273, 4773),
            (17359, 11936, 1620),
            (15004, 68174, 2834),
            (6021, 42389, 2165),
            (3268, 4784, 1483),
            (6954, 12955, 858),
            (12016, 53291, 1809),
            (14749, 76014, 389),
        ),
        start=1,
    )
)


def _usecols(path: Path) -> list[str]:
    header = set(pd.read_csv(path, nrows=0).columns)
    cols = ["id", "time", "xloc_kf", "yloc_kf", "run_id"]
    for extra in ("length_smoothed", "width_smoothed", "keep_ego", "class"):
        if extra in header:
            cols.append(extra)
    return cols


def load_rings(site: str, run_id: int | None = None) -> list[np.ndarray]:
    if site == "jounieh":
        bound = pd.read_csv(ROOT / "data/Lebanon_Jounieh/Jounieh_Road_Boundaries.csv")
        rings = []
        for (kind, pid), g in bound.groupby(["kind", "polygon_id"], sort=True):
            xy = g.sort_values("vertex_index")[["x", "y"]].to_numpy(float)
            if len(xy) and not np.allclose(xy[0], xy[-1]):
                xy = np.vstack([xy, xy[0]])
            rings.append(xy)
        return rings
    if site == "tgsim":
        bound = pd.read_csv(ROOT / "data/TGSIM FB/derived_boundaries/street_boundaries.csv")
        rings = []
        for _, g in bound.groupby("part_index", sort=True):
            xy = g.sort_values("vertex_index")[["x", "y"]].to_numpy(float)
            if len(xy) and not np.allclose(xy[0], xy[-1]):
                xy = np.vstack([xy, xy[0]])
            rings.append(xy)
        return rings
    bound = pd.read_csv(ROOT / "data/Lebanon_Highway/derived_highway_boundaries/highway_boundaries.csv")
    if run_id is not None and "run_id" in bound.columns:
        bound = bound[bound["run_id"].astype(int) == int(run_id)]
    rings = []
    for _, g in bound.groupby(["run_id", "lane_kf"], sort=True):
        g = g.sort_values("point_index")
        rings.append(g[["lower_x", "lower_y"]].to_numpy(float))
        rings.append(g[["upper_x", "upper_y"]].to_numpy(float))
    return rings


def load_tgsim_lane_lines() -> list[np.ndarray]:
    from plot_ref_images import load_tgsim, load_tgsim_lanes
    from shapely.ops import unary_union

    road = load_tgsim()
    curb = road.boundary.buffer(0.45)
    parts = []
    for lane in load_tgsim_lanes():
        edge = lane.boundary.difference(curb)
        if not edge.is_empty:
            parts.append(edge)
    if not parts:
        return []
    geom = unary_union(parts)
    lines = []

    def _collect(item):
        if item is None or item.is_empty:
            return
        if item.geom_type == "LineString":
            xy = np.asarray(item.coords, float)
            if len(xy) >= 2:
                lines.append(xy)
            return
        if hasattr(item, "geoms"):
            for part in item.geoms:
                _collect(part)

    _collect(geom)
    return lines


def _pick_track(spec: dict, *, traj=None, nll=None, scene_cache=None) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    if traj is None:
        traj = pd.read_csv(spec["csv"], usecols=_usecols(spec["csv"]))
    if nll is None:
        nll = pd.read_csv(spec["nll"])
    ego_id = int(spec["ego_id"])
    hit = nll[nll["id"].astype(int) == ego_id]
    if not hit.empty:
        run_id = int(hit.iloc[0]["run_id"])
    else:
        sub = traj[traj["id"].astype(int) == ego_id]
        if sub.empty:
            raise SystemExit(f"{spec['label']}: ego id {ego_id} not in trajectories")
        dur = sub.groupby(sub["run_id"].astype(int))["time"].agg(lambda s: float(s.max()) - float(s.min()))
        run_id = int(dur.idxmax())
    cache_key = (spec["site"], run_id)
    if scene_cache is not None and cache_key in scene_cache:
        scene = scene_cache[cache_key]
    else:
        scene = traj[traj["run_id"].astype(int) == run_id].copy()
        if scene_cache is not None:
            scene_cache[cache_key] = scene
    ego = scene[scene["id"].astype(int) == ego_id].sort_values("time")
    if len(ego) < 8:
        raise SystemExit(f"{spec['label']}: too few samples for id {ego_id}")
    return scene, ego, run_id, ego_id


def _window_start(ego: pd.DataFrame, spec: dict) -> float:
    times = ego["time"].to_numpy(float)
    t_min = float(times.min())
    t_max = float(times.max())
    if spec.get("t0_abs") is not None:
        t0 = min(max(float(spec["t0_abs"]), t_min), t_max - TIMES_S[-1])
        if t0 + TIMES_S[-1] <= t_max + 0.25:
            return t0
    focus = spec.get("focus_xy")
    if focus is not None:
        xy = ego[["xloc_kf", "yloc_kf"]].to_numpy(float)
        d = np.hypot(xy[:, 0] - float(focus[0]), xy[:, 1] - float(focus[1]))
        ok = times + TIMES_S[-1] <= t_max + 0.05
        if not np.any(ok):
            ok = np.ones(len(times), dtype=bool)
        t0 = float(times[ok][int(np.argmin(d[ok]))])
        if t0 - TRAIL_S >= t_min - 0.05 or t0 + TIMES_S[-1] <= t_max + 0.05:
            return t0
    offset = spec.get("t0_from_start")
    if offset is not None:
        t0 = t_min + float(offset)
        if t0 + TIMES_S[-1] <= t_max + 0.05:
            return t0
    need = TRAIL_S + TIMES_S[-1]
    if t_max - t_min >= need + 0.2:
        return t_min + TRAIL_S
    if t_max - t_min >= TIMES_S[-1] + 0.2:
        return t_min
    raise SystemExit(f"ego duration {t_max - t_min:.1f}s is shorter than 6 s")


def _fixed_view(ego: pd.DataFrame, t0: float, spec: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    focus = spec.get("focus_xy")
    if spec.get("pin_focus") and focus is not None:
        cx, cy = float(focus[0]), float(focus[1])
        half = float(spec["view_half"])
        return (cx - half, cx + half), (cy - half, cy + half)
    t_hi = float(t0) + TIMES_S[-1]
    if spec["site"] == "freeway":
        t_lo = float(t0)
        pad = 8.0
    else:
        t_lo = float(t0) - TRAIL_S
        pad = 12.0
    m = (ego["time"] >= t_lo - 1e-9) & (ego["time"] <= t_hi + 1e-9)
    xy = ego.loc[m, ["xloc_kf", "yloc_kf"]].to_numpy(float)
    if len(xy) < 1:
        row = ego.iloc[0]
        xy = np.array([[float(row["xloc_kf"]), float(row["yloc_kf"])]], float)
    xmin, ymin = (float(xy[:, 0].min()), float(xy[:, 1].min()))
    xmax, ymax = (float(xy[:, 0].max()), float(xy[:, 1].max()))
    focus = spec.get("focus_xy")
    if focus is not None:
        xmin = min(xmin, float(focus[0]))
        xmax = max(xmax, float(focus[0]))
        ymin = min(ymin, float(focus[1]))
        ymax = max(ymax, float(focus[1]))
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * max(xmax - xmin, ymax - ymin)
    half = max(half + pad, float(spec["view_half"]))
    return (cx - half, cx + half), (cy - half, cy + half)


def _at_time(g: pd.DataFrame, t: float):
    times = g["time"].to_numpy(float)
    i = int(np.argmin(np.abs(times - t)))
    if abs(times[i] - t) > 0.25:
        return None
    return g.iloc[i]


def _trail(g: pd.DataFrame, t: float) -> np.ndarray:
    m = (g["time"] <= t + 1e-9) & (g["time"] >= t - TRAIL_S - 1e-9)
    return g.loc[m, ["xloc_kf", "yloc_kf"]].to_numpy(float)


def _heading_at(g: pd.DataFrame, t: float) -> float | None:
    times = g["time"].to_numpy(float)
    xy = g[["xloc_kf", "yloc_kf"]].to_numpy(float)
    if len(xy) < 2:
        return None
    i = int(np.argmin(np.abs(times - t)))
    for j in range(i, -1, -1):
        if times[i] - times[j] > 8.0:
            break
        d = xy[i] - xy[j]
        if float(np.hypot(*d)) > 1.0:
            return float(np.arctan2(d[1], d[0]))
    for j in range(i, len(xy)):
        if times[j] - times[i] > 8.0:
            break
        d = xy[j] - xy[i]
        if float(np.hypot(*d)) > 1.0:
            return float(np.arctan2(d[1], d[0]))
    trail = _trail(g, t)
    if len(trail) < 2:
        return None
    d = trail[-1] - trail[0]
    if float(np.hypot(*d)) > 0.35:
        return float(np.arctan2(d[1], d[0]))
    return None


def _is_passenger(row, site: str) -> bool:
    cls = float(row["class"]) if "class" in row.index and pd.notna(row["class"]) else None
    if site == "tgsim":
        return cls in TGSIM_CAR_CLASS
    if site == "freeway":
        return cls == 2.0
    return False


def _size(row, default: tuple[float, float], site: str) -> tuple[float, float]:
    if site == "jounieh":
        return default
    length, width = default
    if "length_smoothed" in row.index and pd.notna(row["length_smoothed"]):
        length = float(row["length_smoothed"])
    if "width_smoothed" in row.index and pd.notna(row["width_smoothed"]):
        width = float(row["width_smoothed"])
    if site in ("freeway", "tgsim") and _is_passenger(row, site):
        length = float(np.clip(length, CAR_LENGTH_M[0], CAR_LENGTH_M[1]))
        width = float(np.clip(width, CAR_WIDTH_M[0], CAR_WIDTH_M[1]))
        return length, width
    return max(length, 1.2), max(width, 0.5)


def _keep_agent(spec: dict, row, is_ego: bool) -> bool:
    if is_ego or spec["site"] != "tgsim":
        return True
    cls = float(row["class"]) if "class" in row.index and pd.notna(row["class"]) else -1.0
    return cls not in TGSIM_SKIP_CLASS


def _fade_trail(ax, hist: np.ndarray, color: str, *, lw: float, z: int) -> None:
    hist = np.asarray(hist, float)
    if len(hist) < 2:
        return
    segs = np.stack([hist[:-1], hist[1:]], axis=1)
    n = len(segs)
    colors = np.zeros((n, 4))
    colors[:, :3] = to_rgba(color)[:3]
    colors[:, 3] = np.linspace(0.12, 0.55, n)
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, capstyle="round", zorder=z))


def _car(ax, pos, heading, length, width, color, z, *, ego: bool) -> None:
    corners = oriented_box_corners(np.asarray(pos, float), float(heading), length, width)
    ax.fill(
        corners[:, 0],
        corners[:, 1],
        facecolor=color,
        edgecolor="none",
        alpha=0.92 if ego else 0.75,
        zorder=z,
        clip_on=True,
    )


def _plot_xy(ax, xy: np.ndarray, **kwargs) -> None:
    xy = np.asarray(xy, float)
    if len(xy) >= 2:
        ax.plot(xy[:, 0], xy[:, 1], **kwargs)


def _scale_bar(ax, length_m: float) -> None:
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x, span_y = x1 - x0, y1 - y0
    x = x0 + 0.06 * span_x
    y = y0 + 0.07 * span_y
    tick = 0.014 * span_y
    ax.plot([x, x + length_m], [y, y], color=INK, lw=1.8, zorder=12, solid_capstyle="butt")
    ax.plot([x, x], [y - tick, y + tick], color=INK, lw=1.5, zorder=12)
    ax.plot([x + length_m, x + length_m], [y - tick, y + tick], color=INK, lw=1.5, zorder=12)
    ax.text(x + 0.5 * length_m, y + 0.03 * span_y, f"{int(length_m)} m", ha="center", va="bottom", fontsize=7.5, color=INK, zorder=12)


def _save_image(fig, path: Path, **kwargs) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(tmp, **kwargs)
    try:
        tmp.replace(path)
        return path
    except OSError:
        alt = path.with_name(path.stem + "_new" + path.suffix)
        tmp.replace(alt)
        return alt


def plot(
    specs: tuple[dict, ...] | None = None,
    *,
    out_name: str = OUT_NAME,
    out_dir: Path | None = None,
    copy_paper: bool = True,
    save_pdf: bool = True,
    tgsim_lanes=None,
    subtitle: str | None = None,
    data: dict | None = None,
    scene_cache: dict | None = None,
    rings_cache: dict | None = None,
    by_id_cache: dict | None = None,
) -> Path:
    specs = tuple(dict(s) for s in (specs or SHOWCASE))
    data = data or {}
    scene_cache = {} if scene_cache is None else scene_cache
    rings_cache = {} if rings_cache is None else rings_cache
    by_id_cache = {} if by_id_cache is None else by_id_cache
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8.5, "axes.titlesize": 10})
    nrows = len(specs)
    fig, axes = plt.subplots(
        nrows,
        3,
        figsize=(10.8, 3.6 if nrows == 1 else 10.2),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(nrows, 3)
    if tgsim_lanes is None:
        tgsim_lanes = load_tgsim_lane_lines()

    for r, spec in enumerate(specs):
        bundle = data.get(spec["site"], {})
        scene, ego, run_id, ego_id = _pick_track(
            spec,
            traj=bundle.get("traj"),
            nll=bundle.get("nll"),
            scene_cache=scene_cache,
        )
        t0 = _window_start(ego, spec)
        bid_key = (spec["site"], run_id)
        if bid_key not in by_id_cache:
            by_id_cache[bid_key] = {int(i): g.sort_values("time") for i, g in scene.groupby("id", sort=False)}
        by_id = by_id_cache[bid_key]
        if bid_key not in rings_cache:
            rings_cache[bid_key] = load_rings(spec["site"], run_id=run_id)
        rings = rings_cache[bid_key]
        xlim, ylim = _fixed_view(ego, t0, spec)
        print(f"{spec['label']}: run {run_id} id {ego_id}  t0={t0:.2f}s", flush=True)

        for c, rel in enumerate(TIMES_S):
            ax = axes[r, c]
            t = t0 + rel
            ax.set_facecolor("white")
            for xy in rings:
                _plot_xy(ax, xy, color=BOUND, lw=0.85, zorder=1, solid_joinstyle="round")
            if spec["site"] == "tgsim":
                for xy in tgsim_lanes:
                    _plot_xy(ax, xy, color=LANE, lw=0.55, ls=(0, (2.8, 2.2)), alpha=0.35, zorder=1)
            for vid, g in by_id.items():
                row = _at_time(g, t)
                if row is None:
                    continue
                is_ego = vid == ego_id
                if not _keep_agent(spec, row, is_ego):
                    continue
                pos = np.array([float(row["xloc_kf"]), float(row["yloc_kf"])], float)
                if not (xlim[0] - 8 <= pos[0] <= xlim[1] + 8 and ylim[0] - 8 <= pos[1] <= ylim[1] + 8):
                    continue
                trail = _trail(g, t)
                yaw = _heading_at(g, t)
                if yaw is None and not is_ego:
                    continue
                if yaw is None:
                    yaw = 0.0
                color = EGO if is_ego else OTHER
                _fade_trail(ax, trail, color, lw=0.7 if is_ego else 0.45, z=2 if is_ego else 1)
                length, width = _size(row, spec["default_size"], spec["site"])
                _car(ax, pos, yaw, length, width, color, z=4 if is_ego else 3, ego=is_ego)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("#d5dbe3")
                spine.set_linewidth(0.8)
            if r == 0:
                ax.set_title(f"$t = {int(rel)}$ s", pad=6)
            if c == 0:
                ax.set_ylabel(spec["label"], fontsize=11, fontweight="semibold", color=INK, labelpad=8)
                _scale_bar(ax, spec["scale_m"])

    handles = [
        Line2D([0], [0], color=EGO, lw=3, label="ego"),
        Line2D([0], [0], color=OTHER, lw=3, label="others"),
        Line2D([0], [0], color=OTHER, lw=1.2, label="3 s tail"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    if subtitle:
        fig.suptitle(subtitle, fontsize=9, color="#555555", y=1.06)

    dest = out_dir or OUTPUTS[0]
    dest.mkdir(parents=True, exist_ok=True)
    png = _save_image(fig, dest / f"{out_name}.png", dpi=280, bbox_inches="tight", facecolor="white")
    if save_pdf:
        _save_image(fig, dest / f"{out_name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if copy_paper:
        payload = png.read_bytes()
        pdf_bytes = (dest / f"{out_name}.pdf").read_bytes() if save_pdf else None
        for folder in OUTPUTS[1:]:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / png.name).write_bytes(payload)
            if pdf_bytes is not None:
                (folder / f"{out_name}.pdf").write_bytes(pdf_bytes)
    return png


def _specs_for(ids: dict[str, int]) -> tuple[dict, ...]:
    specs = []
    for spec in SHOWCASE:
        row = dict(spec)
        row["ego_id"] = int(ids[spec["site"]])
        specs.append(row)
    return tuple(specs)


def plot_variants() -> list[Path]:
    lanes = load_tgsim_lane_lines()
    data = {
        spec["site"]: {
            "traj": pd.read_csv(spec["csv"], usecols=_usecols(spec["csv"])),
            "nll": pd.read_csv(spec["nll"]),
        }
        for spec in SHOWCASE
    }
    scene_cache: dict = {}
    rings_cache: dict = {}
    by_id_cache: dict = {}
    paths = []
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    for var in VARIANTS:
        ids = var["ids"]
        subtitle = (
            f"candidate {var['tag']}  ·  Highway {ids['freeway']}  ·  "
            f"Jounieh {ids['jounieh']}  ·  TGSIM {ids['tgsim']}"
        )
        try:
            path = plot(
                _specs_for(ids),
                out_name=f"all_sites_snapshots_{var['tag']}",
                out_dir=CANDIDATE_DIR,
            copy_paper=False,
            save_pdf=False,
            tgsim_lanes=lanes,
                subtitle=subtitle,
                data=data,
                scene_cache=scene_cache,
                rings_cache=rings_cache,
                by_id_cache=by_id_cache,
            )
        except SystemExit as exc:
            print(f"skip {var['tag']}: {exc}", flush=True)
            continue
        paths.append(path)
    if paths:
        from matplotlib.image import imread

        n = len(paths)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, path in zip(axes, paths):
            ax.imshow(imread(path))
            ax.set_title(path.stem.replace("all_sites_snapshots_", "cand "), fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in axes[n:]:
            ax.set_axis_off()
        fig.tight_layout()
        sheet = CANDIDATE_DIR / "all_sites_snapshots_contact_sheet.png"
        fig.savefig(sheet, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(sheet, flush=True)
    return paths


def plot_paper() -> Path:
    return plot(copy_paper=True, save_pdf=True)


TGSIM_ALTS = (
    {"tag": "T01", "ego_id": 2485, "t0_abs": 6316.0, "focus_xy": (39.0, 112.0), "where": "main"},
    {"tag": "T02", "ego_id": 2541, "t0_abs": 6422.0, "focus_xy": (39.0, 112.0), "where": "main"},
    {"tag": "T03", "ego_id": 2534, "t0_abs": 6419.0, "focus_xy": (39.0, 112.0), "where": "main"},
    {"tag": "T04", "ego_id": 2503, "t0_abs": 6318.0, "focus_xy": (39.0, 112.0), "where": "main"},
    {"tag": "T05", "ego_id": 2493, "t0_abs": 6318.0, "focus_xy": (39.0, 112.0), "where": "main"},
    {"tag": "T06", "ego_id": 3159, "t0_abs": 6318.0, "focus_xy": (39.0, 112.0), "where": "main (truck)"},
    {"tag": "T07", "ego_id": 1482, "t0_abs": 3210.0, "focus_xy": (38.0, 190.0), "where": "north T"},
    {"tag": "T08", "ego_id": 1489, "t0_abs": 3210.0, "focus_xy": (38.0, 190.0), "where": "north T"},
    {"tag": "T09", "ego_id": 2980, "t0_abs": 3210.0, "focus_xy": (38.0, 190.0), "where": "north T"},
    {"tag": "T10", "ego_id": 2548, "t0_abs": 6471.0, "focus_xy": (40.0, 55.0), "where": "south"},
    {"tag": "T11", "ego_id": 2561, "t0_abs": 6471.0, "focus_xy": (40.0, 55.0), "where": "south"},
    {"tag": "T12", "ego_id": 2562, "t0_abs": 6471.0, "focus_xy": (40.0, 55.0), "where": "south"},
)


def plot_tgsim_alts() -> list[Path]:
    lanes = load_tgsim_lane_lines()
    tgsim = SHOWCASE[2]
    data = {
        "tgsim": {
            "traj": pd.read_csv(tgsim["csv"], usecols=_usecols(tgsim["csv"])),
            "nll": pd.read_csv(tgsim["nll"]),
        }
    }
    scene_cache: dict = {}
    rings_cache: dict = {}
    by_id_cache: dict = {}
    out_dir = CANDIDATE_DIR / "tgsim_dense"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for var in TGSIM_ALTS:
        spec = dict(tgsim)
        spec["ego_id"] = int(var["ego_id"])
        spec["t0_abs"] = float(var["t0_abs"])
        spec["focus_xy"] = var["focus_xy"]
        spec["pin_focus"] = True
        spec["view_half"] = 30.0
        spec["scale_m"] = 20
        subtitle = f"{var['tag']}  ·  TGSIM id {var['ego_id']}  ·  {var['where']}"
        try:
            path = plot(
                (spec,),
                out_name=f"tgsim_dense_{var['tag']}",
                out_dir=out_dir,
                copy_paper=False,
                save_pdf=False,
                tgsim_lanes=lanes,
                subtitle=subtitle,
                data=data,
                scene_cache=scene_cache,
                rings_cache=rings_cache,
                by_id_cache=by_id_cache,
            )
        except SystemExit as exc:
            print(f"skip {var['tag']}: {exc}", flush=True)
            continue
        paths.append(path)
    if paths:
        from matplotlib.image import imread

        n = len(paths)
        cols = 3
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(12.6, 3.4 * rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, path in zip(axes, paths):
            ax.imshow(imread(path))
            ax.set_title(path.stem.replace("tgsim_dense_", ""), fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
        for ax in axes[n:]:
            ax.set_axis_off()
        fig.tight_layout()
        sheet = out_dir / "tgsim_dense_contact_sheet.png"
        fig.savefig(sheet, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(sheet, flush=True)
    return paths


if __name__ == "__main__":
    if "--variants" in sys.argv:
        for p in plot_variants():
            print(p)
    elif "--tgsim-alts" in sys.argv:
        for p in plot_tgsim_alts():
            print(p)
    else:
        print(plot_paper())

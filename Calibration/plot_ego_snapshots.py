#!/usr/bin/env python
"""Six-panel ego snapshot sequence with short past trails.

One good low-NLL ego per site (freeway / jounieh / tgsim). Draws 6 frames at
t = T/6, 2T/6, ..., T. Ego is red; other vehicles are gray. Each marker has a
thin trail of the past ``trail_sec`` seconds.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Showcase IDs: low NLL + enough travel for readable snapshots.
DEFAULT_IDS = {
    "freeway": 7346,
    "jounieh": 12549,
    "tgsim": 1745,
}


def site_paths(site: str) -> tuple[Path, Path, Path]:
    if site == "freeway":
        return (
            ROOT / "data/Lebanon_Highway/Final_Lebanon_Data.csv",
            ROOT / "Calibration/diagnostics/per_id/per_id_best90_by_nll.csv",
            ROOT / "Calibration/diagnostics/per_id",
        )
    if site == "jounieh":
        return (
            ROOT / "data/Lebanon_Jounieh/prepared/trajectories_calibration.csv",
            ROOT / "Calibration/diagnostics_jounieh/per_id/per_id_best90_by_nll.csv",
            ROOT / "Calibration/diagnostics_jounieh/per_id",
        )
    if site == "tgsim":
        return (
            ROOT / "data/TGSIM FB/prepared/trajectories_calibration.csv",
            ROOT / "Calibration/diagnostics_tgsim/per_id/per_id_best90_by_nll.csv",
            ROOT / "Calibration/diagnostics_tgsim/per_id",
        )
    raise ValueError(site)


def load_rings(site: str, run_id: int | None = None) -> list[np.ndarray]:
    if site == "jounieh":
        bound = pd.read_csv(ROOT / "data/Lebanon_Jounieh/Jounieh_Road_Boundaries.csv")
        rings = []
        for _, g in bound.groupby(["kind", "polygon_id"], sort=True):
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
    if site == "freeway":
        bound = pd.read_csv(
            ROOT / "data/Lebanon_Highway/derived_highway_boundaries/highway_boundaries.csv"
        )
        if run_id is not None and "run_id" in bound.columns:
            bound = bound[bound["run_id"].astype(int) == int(run_id)]
        rings = []
        for _, g in bound.groupby(["run_id", "lane_kf"], sort=True):
            g = g.sort_values("point_index")
            lo = g[["lower_x", "lower_y"]].to_numpy(float)
            up = g[["upper_x", "upper_y"]].to_numpy(float)
            rings.append(lo)
            rings.append(up)
        return rings
    return []


def pick_good_id(
    traj: pd.DataFrame,
    nll: pd.DataFrame,
    *,
    min_dur: float = 12.0,
    min_disp: float = 15.0,
    prefer_id: int | None = None,
) -> pd.Series:
    if prefer_id is not None:
        hit = nll[nll["id"].astype(int) == int(prefer_id)]
        if hit.empty:
            raise SystemExit(f"prefer_id={prefer_id} not in best90 table")
        return hit.iloc[0]

    ranked = nll.sort_values("local_nll").head(80)
    cand = {(int(r.run_id), int(r.id)) for r in ranked.itertuples()}
    rid = traj["run_id"].to_numpy(int)
    iid = traj["id"].to_numpy(int)
    mask = np.zeros(len(traj), dtype=bool)
    for run, vid in cand:
        mask |= (rid == run) & (iid == vid)
    sub = traj.loc[mask]
    rows = []
    has_keep = "keep_ego" in sub.columns
    for (run, vid), g in sub.groupby(["run_id", "id"], sort=False):
        if has_keep and not bool(g["keep_ego"].iloc[0]):
            continue
        t0, t1 = float(g["time"].min()), float(g["time"].max())
        dur = t1 - t0
        if dur < min_dur:
            continue
        x0, y0 = float(g["xloc_kf"].iloc[0]), float(g["yloc_kf"].iloc[0])
        x1, y1 = float(g["xloc_kf"].iloc[-1]), float(g["yloc_kf"].iloc[-1])
        disp = float(np.hypot(x1 - x0, y1 - y0))
        if disp < min_disp:
            continue
        rows.append({"run_id": int(run), "id": int(vid), "dur": dur, "disp": disp})
    meta = pd.DataFrame(rows).merge(
        ranked[["run_id", "id", "local_nll"]],
        on=["run_id", "id"],
    )
    if meta.empty:
        raise SystemExit("No ID met duration/displacement filters among top NLL")
    top = meta.nsmallest(min(15, len(meta)), "local_nll")
    return top.sort_values(["disp", "dur"], ascending=False).iloc[0]


def heading_at(xy: np.ndarray) -> float:
    if len(xy) < 2:
        return 0.0
    d = xy[-1] - xy[-2]
    if np.hypot(*d) < 1e-6 and len(xy) >= 3:
        d = xy[-1] - xy[-3]
    if np.hypot(*d) < 1e-6:
        return 0.0
    return float(np.arctan2(d[1], d[0]))


def draw_vehicle(
    ax,
    x: float,
    y: float,
    yaw: float,
    length: float,
    width: float,
    *,
    color: str,
    zorder: int,
) -> None:
    length = max(float(length), 2.0)
    width = max(float(width), 1.0)
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array(
        [
            [-length / 2, -width / 2],
            [length / 2, -width / 2],
            [length / 2, width / 2],
            [-length / 2, width / 2],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    world = corners @ rot.T + np.array([x, y])
    ax.fill(
        world[:, 0],
        world[:, 1],
        facecolor=color,
        edgecolor="none",
        alpha=0.92 if color == "#C0392B" else 0.75,
        zorder=zorder,
    )


def nearest_rows(df: pd.DataFrame, t: float, tol: float = 0.12) -> pd.DataFrame:
    times = df["time"].to_numpy(float)
    hit = df[np.isclose(times, t, atol=1e-6)]
    if not hit.empty:
        return hit
    out = []
    for _, g in df.groupby("id", sort=False):
        i = int(np.argmin(np.abs(g["time"].to_numpy(float) - t)))
        row = g.iloc[i]
        if abs(float(row["time"]) - t) <= tol:
            out.append(row)
    return pd.DataFrame(out) if out else df.iloc[0:0]


def trail_for(g: pd.DataFrame, t: float, trail_sec: float) -> np.ndarray:
    m = (g["time"] <= t + 1e-9) & (g["time"] >= t - trail_sec - 1e-9)
    return g.loc[m, ["xloc_kf", "yloc_kf"]].to_numpy(float)


def plot_snapshots(
    site: str,
    *,
    prefer_id: int | None = None,
    trail_sec: float = 2.0,
    view_half: float | None = None,
    n_frames: int = 6,
    min_disp: float | None = None,
) -> Path:
    if view_half is None:
        view_half = {"freeway": 70.0, "jounieh": 96.0, "tgsim": 45.0}[site]
    if min_disp is None:
        min_disp = 40.0 if site == "freeway" else 15.0

    traj_path, nll_path, out_dir = site_paths(site)
    header = set(pd.read_csv(traj_path, nrows=0).columns)
    usecols = ["id", "time", "xloc_kf", "yloc_kf", "speed_kf", "run_id"]
    for extra in ("length_smoothed", "width_smoothed", "keep_ego"):
        if extra in header:
            usecols.append(extra)

    traj = pd.read_csv(traj_path, usecols=usecols)
    nll = pd.read_csv(nll_path)
    pick = pick_good_id(
        traj, nll, prefer_id=prefer_id, min_disp=min_disp
    )
    run_id = int(pick["run_id"])
    ego_id = int(pick["id"])
    local_nll = float(pick["local_nll"])

    scene = traj[traj["run_id"] == run_id].copy()
    ego = scene[scene["id"] == ego_id].sort_values("time")
    if ego.empty:
        raise SystemExit(f"No trajectory for run={run_id} id={ego_id}")
    t0 = float(ego["time"].min())
    t1 = float(ego["time"].max())
    frame_abs = np.linspace(t0 + (t1 - t0) / n_frames, t1, n_frames)
    frame_rel = frame_abs - t0

    by_id = {int(i): g.sort_values("time") for i, g in scene.groupby("id", sort=False)}
    rings = load_rings(site, run_id=run_id)
    site_label = {"freeway": "FREEWAY", "jounieh": "JOUNIEH", "tgsim": "TGSIM"}[site]

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    fig.suptitle(
        f"{site_label}  ·  ego id={ego_id}  ·  NLL={local_nll:.3f}  ·  "
        f"duration={t1 - t0:.1f}s  ·  trail={trail_sec:.0f}s",
        fontsize=12,
    )

    for ax, t_abs, t_rel in zip(axes.ravel(), frame_abs, frame_rel):
        for xy in rings:
            ax.plot(xy[:, 0], xy[:, 1], color="#9AA5B1", lw=0.8, zorder=0)

        snap = nearest_rows(scene, float(t_abs))
        if snap.empty:
            ax.set_title(f"t = {t_rel:.1f}s (no data)")
            ax.set_aspect("equal")
            continue

        ego_row = snap[snap["id"] == ego_id]
        if ego_row.empty:
            eg = ego.iloc[int(np.argmin(np.abs(ego["time"].to_numpy(float) - t_abs)))]
            ex, ey = float(eg["xloc_kf"]), float(eg["yloc_kf"])
        else:
            eg = ego_row.iloc[0]
            ex, ey = float(eg["xloc_kf"]), float(eg["yloc_kf"])

        for _, row in snap.iterrows():
            vid = int(row["id"])
            is_ego = vid == ego_id
            color = "#C0392B" if is_ego else "#A0A0A0"
            g = by_id[vid]
            trail = trail_for(g, float(t_abs), trail_sec)
            if len(trail) >= 2:
                ax.plot(
                    trail[:, 0],
                    trail[:, 1],
                    color=color,
                    lw=0.7 if is_ego else 0.45,
                    alpha=0.55 if is_ego else 0.35,
                    solid_capstyle="round",
                    zorder=2 if is_ego else 1,
                )
            yaw = heading_at(trail) if len(trail) else 0.0
            if site == "jounieh":
                # These source columns are image-axis extents, not the
                # oriented vehicle footprint used in the simulator.
                length, width = 4.5, 1.8
            else:
                length = float(row["length_smoothed"]) if "length_smoothed" in row.index and pd.notna(row["length_smoothed"]) else 4.5
                width = float(row["width_smoothed"]) if "width_smoothed" in row.index and pd.notna(row["width_smoothed"]) else 1.8
            draw_vehicle(
                ax,
                float(row["xloc_kf"]),
                float(row["yloc_kf"]),
                yaw,
                length,
                width,
                color=color,
                zorder=4 if is_ego else 3,
            )

        ax.set_xlim(ex - view_half, ex + view_half)
        ax.set_ylim(ey - view_half, ey + view_half)
        ax.set_aspect("equal")
        ax.set_title(f"t = {t_rel:.1f} s", fontsize=11)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)

    axes.ravel()[0].plot([], [], color="#C0392B", lw=2, label="ego")
    axes.ravel()[0].plot([], [], color="#A0A0A0", lw=2, label="others")
    axes.ravel()[0].legend(loc="upper right", fontsize=8, framealpha=0.9)

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"run_{run_id:02d}_id_{ego_id:06d}_snapshots_6panel.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--site",
        choices=("freeway", "jounieh", "tgsim", "all"),
        default="all",
    )
    ap.add_argument("--id", type=int, default=None, help="Force a specific ego id")
    ap.add_argument("--trail-sec", type=float, default=2.0)
    ap.add_argument("--view-half", type=float, default=None)
    ap.add_argument(
        "--auto",
        action="store_true",
        help="Auto-pick ID instead of the showcase default",
    )
    args = ap.parse_args()
    sites = ["freeway", "jounieh", "tgsim"] if args.site == "all" else [args.site]
    for site in sites:
        if args.auto:
            prefer = None
        elif args.id is not None and args.site != "all":
            prefer = args.id
        else:
            prefer = DEFAULT_IDS[site]
        plot_snapshots(
            site,
            prefer_id=prefer,
            trail_sec=args.trail_sec,
            view_half=args.view_half,
        )


if __name__ == "__main__":
    main()

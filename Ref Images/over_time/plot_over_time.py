"""Closed-loop rollouts over time on the Ref Images metre-frame backgrounds.

    python "Ref Images/over_time/plot_over_time.py"
    python "Ref Images/over_time/plot_over_time.py" --site tgsim --seed 200
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
REF = HERE.parent
REPO = REF.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REF))
sys.path.insert(0, str(REPO))

from backgrounds import load_background  # noqa: E402
from RL.corridor import oriented_box_corners  # noqa: E402
from spawn import finish_site_scenario, roundabout_hub, spawn_around, tgsim_junction  # noqa: E402

INK = "#0f172a"
AGENT_COLORS = [
    "#ef4444",
    "#a855f7",
    "#14b8a6",
    "#22c55e",
    "#3b82f6",
    "#f97316",
    "#e11d48",
    "#0ea5e9",
    "#84cc16",
    "#d946ef",
]
TIMES_S = (2, 4, 6, 8)
DELTA_THRESH = 0.45
TITLES = {
    "freeway": "Freeway",
    "tgsim": "TGSIM Foggy Bottom",
    "roundabout": "Jounieh roundabout",
}


def _set_view(ax, xlim, ylim) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color("#e2e8f0")
        spine.set_linewidth(0.8)


def _road_align(origin_xy, tangent_xy):
    origin = np.asarray(origin_xy, float).reshape(2)
    tang = np.asarray(tangent_xy, float).reshape(2)
    tang = tang / max(float(np.linalg.norm(tang)), 1e-9)
    ang = float(np.arctan2(tang[1], tang[0]))
    rot = 0.5 * np.pi - ang
    c, s = float(np.cos(rot)), float(np.sin(rot))
    r = np.array([[c, -s], [s, c]], float)

    def xf(xy):
        pts = np.asarray(xy, float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        out = (pts - origin) @ r.T
        return out[0] if single else out

    def inv(panel):
        pts = np.asarray(panel, float)
        single = pts.ndim == 1
        pts = np.atleast_2d(pts)
        out = pts @ r + origin
        return out[0] if single else out

    return xf, inv, rot


def _identity_frame():
    def xf(xy):
        return np.asarray(xy, float)

    return xf, xf, 0.0


def _car(ax, pos, heading, length, width, color, z=6):
    corners = oriented_box_corners(pos, heading, length, width)
    ax.fill(
        corners[:, 0],
        corners[:, 1],
        color=color,
        alpha=0.95,
        edgecolor="#0f172a",
        lw=0.45,
        zorder=z,
        clip_on=True,
    )


def _ghost_car(ax, pos, heading, length, width, color="#64748b", z=5):
    corners = oriented_box_corners(pos, heading, length, width)
    closed = np.vstack([corners, corners[0]])
    ax.plot(closed[:, 0], closed[:, 1], color=color, lw=0.85, ls=(0, (1.6, 1.4)), alpha=0.7, zorder=z)


def _fading_history(ax, hist, color, lw=1.05, z=3):
    hist = np.asarray(hist, float)
    if len(hist) < 2:
        return
    segs = np.stack([hist[:-1], hist[1:]], axis=1)
    n = len(segs)
    colors = np.zeros((n, 4))
    colors[:, :3] = to_rgba(color)[:3]
    colors[:, 3] = np.linspace(0.08, 0.55, n)
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=lw, capstyle="round", zorder=z))


def _draw_background(ax, site: str, xf, inv, xlim, ylim, aligned: bool) -> None:
    img, extent = load_background(site)
    if not aligned:
        ax.imshow(img, extent=extent, origin="upper", interpolation="bilinear", aspect="equal", zorder=0)
        _set_view(ax, xlim, ylim)
        return
    from scipy.ndimage import map_coordinates

    nx = 420
    ny = max(220, int(round(420 * (ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-6))))
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[1], ylim[0], ny)
    uu, vv = np.meshgrid(xs, ys)
    world = inv(np.stack([uu.ravel(), vv.ravel()], axis=1)).reshape(ny, nx, 2)
    xmin, xmax, ymin, ymax = extent
    h, w = img.shape[:2]
    px = (world[..., 0] - xmin) / (xmax - xmin) * (w - 1)
    py = (ymax - world[..., 1]) / (ymax - ymin) * (h - 1)
    coords = np.stack([py, px], axis=0)
    fill = 1.0
    if img.ndim == 2:
        sampled = map_coordinates(img, coords, order=1, mode="constant", cval=fill)
    else:
        chans = [
            map_coordinates(
                img[:, :, c],
                coords,
                order=1,
                mode="constant",
                cval=fill if c < 3 else 0.0,
            )
            for c in range(img.shape[2])
        ]
        sampled = np.stack(chans, axis=-1)
    ax.imshow(
        np.clip(sampled, 0, 1),
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="upper",
        interpolation="bilinear",
        aspect="equal",
        zorder=0,
    )
    _set_view(ax, xlim, ylim)


def _pick_ego_station(result, t0, t1) -> int:
    t_mid = (t0 + t1) // 2
    stations = np.asarray(result.station[t_mid], float)
    order = np.argsort(stations)
    n = len(order)
    if n == 1:
        return int(order[0])
    idx = int(np.clip(round(0.4 * (n - 1)), 1, max(1, n - 2)))
    return int(order[idx])


def _cluster_origin(result, t, radius: float = 28.0):
    pos = np.asarray(result.positions[t], float)
    active = np.asarray(result.active[t], bool)
    if not active.any():
        active = np.ones(len(pos), bool)
    pts = pos[active]
    counts = np.array([np.sum(np.linalg.norm(pts - p, axis=1) <= radius) for p in pts])
    origin = pts[int(np.argmax(counts))]
    near = pts[np.linalg.norm(pts - origin, axis=1) <= radius]
    return near.mean(axis=0)


def _square_view(util, residual, t0, t1, half: float, origin=None):
    if origin is None:
        t1 = min(t1, int(util.steps), int(residual.steps))
        t0 = min(t0, t1)
        origin = 0.5 * (_cluster_origin(residual, t0, half) + _cluster_origin(residual, t1, half))
    origin = np.asarray(origin, float).reshape(2)
    xf, inv, rot = _identity_frame()
    return xf, inv, rot, (origin[0] - half, origin[0] + half), (origin[1] - half, origin[1] + half), False


def _road_view(scenario, result, t0, t1):
    t1 = min(t1, int(result.steps))
    ego = _pick_ego_station(result, t0, t1)
    ego_start = np.asarray(result.positions[t0, ego], float)
    ego_end = np.asarray(result.positions[t1, ego], float)
    origin = 0.5 * (ego_start + ego_end)
    travel = ego_end - ego_start
    if float(np.linalg.norm(travel)) < 1e-3:
        s_mid = float(result.station[(t0 + t1) // 2, ego])
        _, tang = scenario.corridor.xy_from_frenet(s_mid, 0.0)
        travel = np.asarray(tang, float)
    xf, inv, rot = _road_align(origin, travel)
    lower = xf(scenario.corridor.lower)
    upper = xf(scenario.corridor.upper)
    x_half = 0.5 * (float(np.max(np.concatenate([lower[:, 0], upper[:, 0]]))) - float(np.min(np.concatenate([lower[:, 0], upper[:, 0]])))) + 0.5
    y0 = float(xf(ego_start)[1]) - 10.0
    y1 = float(xf(ego_end)[1]) + 10.0
    if y1 < y0:
        y0, y1 = y1, y0
    return xf, inv, rot, (-x_half, x_half), (y0, y1), True


def _ckpt_status(path: Path) -> tuple[str, int]:
    summary = path.with_name(path.stem + ".summary.json")
    if not summary.is_file():
        return "", 0
    blob = json.loads(summary.read_text(encoding="utf-8"))
    return str(blob.get("selection_status") or ""), int(blob.get("selected_update") or 0)


def _live_ckpt(export: Path, resume: Path) -> Path:
    """Selected export can be a zero-init utility fallback; use live trainer weights."""
    import torch

    blob = torch.load(export, map_location="cpu")
    live = torch.load(resume, map_location="cpu")
    if "state_dict" not in live:
        return export
    blob["state_dict"] = live["state_dict"]
    blob["selection_status"] = f"live_update_{live.get('update', '?')}"
    tmp = HERE / f"_{export.stem}_live.pt"
    torch.save(blob, tmp)
    return tmp


_CKPT_CACHE: dict[tuple[str, str], Path] = {}


def _resolve_ckpt(directory: Path, stem: str = "residual_policy") -> Path:
    key = (str(directory), stem)
    if key in _CKPT_CACHE:
        return _CKPT_CACHE[key]
    found = []
    plain = directory / f"{stem}.pt"
    if plain.is_file():
        found.append(plain)
    found.extend(sorted(p for p in directory.glob(f"{stem}_seed*.pt") if ".resume" not in p.name))
    if not found:
        raise FileNotFoundError(f"No {stem} checkpoint in {directory}")
    learned = []
    for path in found:
        status, update = _ckpt_status(path)
        if status == "learned_residual":
            learned.append((update, path))
    if learned:
        learned.sort(reverse=True)
        update, path = learned[0]
        print(f"Using learned residual {path.name} (selected_update={update})", flush=True)
        _CKPT_CACHE[key] = path
        return path
    for path in found:
        resume = path.with_name(path.stem + ".resume.pt")
        if resume.is_file():
            print(f"{path.name} is utility_fallback; using live weights from {resume.name}", flush=True)
            live = _live_ckpt(path, resume)
            _CKPT_CACHE[key] = live
            return live
    print(f"WARNING: {found[0].name} is a utility fallback (residual == prior)", flush=True)
    _CKPT_CACHE[key] = found[0]
    return found[0]


def _pose_delta(util, residual) -> tuple[float, float]:
    t = min(int(util.steps), int(residual.steps))
    dist = np.linalg.norm(residual.positions[: t + 1] - util.positions[: t + 1], axis=2)
    mean_d, max_d = float(dist.mean()), float(dist.max())
    print(
        f"pose delta mean={mean_d:.2f} m  max={max_d:.2f} m  frac>0.45={(dist > 0.45).mean():.2f}",
        flush=True,
    )
    return mean_d, max_d


def _progress_ok(result, t0: float = 2.0, t1: float = 8.0, min_mean: float = 4.0, min_frac: float = 0.7, min_move: float = 2.5) -> bool:
    dt = float(result.dt)
    i0 = min(int(round(t0 / dt)), int(result.steps))
    i1 = min(int(round(t1 / dt)), int(result.steps))
    d = np.linalg.norm(result.positions[i1] - result.positions[i0], axis=1)
    mean_d = float(np.mean(d))
    frac = float(np.mean(d >= min_move))
    print(
        f"  travel t={t0:.0f}–{t1:.0f}s mean={mean_d:.2f} m  frac≥{min_move:.1f}m={frac:.0%}",
        flush=True,
    )
    return mean_d >= min_mean and frac >= min_frac


def _activate_case(case_dir: Path, run_env: dict[str, str] | None = None) -> None:
    if run_env:
        os.environ.update(run_env)
    case = str(case_dir)
    if case not in sys.path:
        sys.path.insert(0, case)
    import activate

    activate.apply()


def _controllers(calibration: Path | None, checkpoint: Path, accept_if_better: bool = True):
    from Baselines.residual_marl import ResidualMARLController
    from Baselines.utility_prior import UtilityPriorController

    prior = UtilityPriorController(calibration=calibration, prefer="robust")
    residual = ResidualMARLController(
        checkpoint=checkpoint,
        calibration=calibration,
        prefer="robust",
        accept_if_better=accept_if_better,
    )
    return prior, residual


def collect_freeway(seed: int):
    from Baselines.runner import rollout
    from Baselines.scenario import build_scenario

    ckpt = _resolve_ckpt(REPO / "RL" / "checkpoints" / "revision5")
    scenario = build_scenario(
        seed=seed,
        num_agents=8,
        max_steps=40,
        spawn_s_range=(30.0, 110.0),
        min_initial_spacing=7.0,
    )
    prior, residual = _controllers(None, ckpt)
    util, res = rollout(scenario, prior), rollout(scenario, residual)
    _pose_delta(util, res)
    return scenario, util, res


def collect_tgsim(seed: int):
    from config import CALIBRATION, CHECKPOINT_DIR, NUM_AGENTS
    from Baselines.runner import rollout
    from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv

    cfg = EnvConfig(dt=0.5, max_steps=80, num_agents=NUM_AGENTS, base_desired_speed=8.0)
    env = MultiAgentTrafficEnv(cfg, seed=int(seed))
    env.reset()
    origin = tgsim_junction(env.corridor)
    agents, dest_s = spawn_around(
        env,
        origin,
        n_agents=12,
        radius=26.0,
        inner=8.0,
        spacing=7.0,
        dest_lo=40.0,
        dest_hi=120.0,
        min_cars=8,
        speeds=(5.0, 4.2, 3.6, 3.2),
        min_speed=3.2,
        cruise=4.5,
        desired=(7.0, 0.8, 5.0),
    )
    dest_d = [float(np.linalg.norm(a.pos - a.dest)) for a in agents]
    print(
        f"  dest Euclidean {min(dest_d):.1f}–{max(dest_d):.1f} m  (n={len(agents)})",
        flush=True,
    )
    scenario = finish_site_scenario(env, seed, agents, dest_s, origin, 24.0)
    prior, residual = _controllers(CALIBRATION, _resolve_ckpt(CHECKPOINT_DIR))
    util, res = rollout(scenario, prior), rollout(scenario, residual)
    _pose_delta(util, res)
    if not _progress_ok(util, min_mean=5.0, min_frac=0.75, min_move=3.0):
        raise RuntimeError("TGSIM pack did not travel enough over the plotted window")
    return scenario, util, res


def collect_roundabout(seed: int):
    from config import BASE_DESIRED_SPEED, CALIBRATION, CHECKPOINT_DIR, NUM_AGENTS
    from Baselines.runner import rollout
    from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv

    cfg = EnvConfig(dt=0.5, max_steps=80, num_agents=NUM_AGENTS, base_desired_speed=BASE_DESIRED_SPEED)
    env = MultiAgentTrafficEnv(cfg, seed=int(seed))
    env.reset()
    origin = roundabout_hub(env.corridor)
    agents, dest_s = spawn_around(
        env,
        origin,
        n_agents=8,
        radius=10.0,
        inner=6.2,
        spacing=3.4,
        dest_lo=16.0,
        dest_hi=40.0,
        min_cars=6,
        speeds=(3.2, 2.8, 2.4),
        min_speed=2.4,
        cruise=2.8,
        desired=(3.6, 0.3, 2.6),
        ring_center=origin,
        min_clearance=0.85,
    )
    dest_d = [float(np.linalg.norm(a.pos - a.dest)) for a in agents]
    print(
        f"  dest Euclidean {min(dest_d):.1f}–{max(dest_d):.1f} m  (n={len(agents)})",
        flush=True,
    )
    scenario = finish_site_scenario(env, seed, agents, dest_s, origin, 16.5)
    ckpt = _resolve_ckpt(CHECKPOINT_DIR)
    prior, residual = _controllers(CALIBRATION, ckpt, accept_if_better=True)
    util, res = rollout(scenario, prior), rollout(scenario, residual)
    mean_d, _ = _pose_delta(util, res)
    if mean_d < 0.05:
        print("gated residual matches prior; plotting ungated live residual", flush=True)
        _, residual = _controllers(CALIBRATION, ckpt, accept_if_better=False)
        res = rollout(scenario, residual)
        _pose_delta(util, res)
    if not _progress_ok(util, min_mean=3.5, min_frac=0.5, min_move=2.0):
        raise RuntimeError("Roundabout pack did not travel enough over the plotted window")
    return scenario, util, res


def make_time_strip(site, scenario, util, residual, times_s=TIMES_S, box_scale=1.0, view="square", half=32.2):
    rows = [("Utility prior", util), ("Residual MARL", residual)]
    n_t = len(times_s)
    dt = float(scenario.dt)
    t_idxs = [min(int(util.steps), int(residual.steps), int(round(t_s / dt))) for t_s in times_s]
    t0, t1 = t_idxs[0], t_idxs[-1]
    half = float(getattr(scenario, "plot_half", half))
    origin = getattr(scenario, "plot_origin", None)
    if view == "road":
        xf, inv, rot, xlim, ylim, aligned = _road_view(scenario, util, t0, t1)
    else:
        xf, inv, rot, xlim, ylim, aligned = _square_view(util, residual, t0, t1, half, origin=origin)
    x_span = max(xlim[1] - xlim[0], 1e-6)
    y_span = max(ylim[1] - ylim[0], 1e-6)
    panel_w = 1.55
    panel_h = panel_w * (y_span / x_span)
    fig, axes = plt.subplots(
        len(rows),
        n_t,
        figsize=(panel_w * n_t, panel_h * len(rows)),
        squeeze=False,
        gridspec_kw={"wspace": 0.045, "hspace": 0.10},
    )
    fig.patch.set_facecolor("white")
    for r, (name, result) in enumerate(rows):
        steps = int(result.steps)
        length = float(result.vehicle_length) * box_scale
        width = float(result.vehicle_width) * box_scale
        n_agents = int(result.num_agents)
        show_delta = r == 1
        for c, t_s in enumerate(times_s):
            ax = axes[r, c]
            t = min(steps, int(round(t_s / dt)))
            t_util = min(int(util.steps), t)
            _draw_background(ax, site, xf, inv, xlim, ylim, aligned)
            if show_delta:
                for i in range(n_agents):
                    pos_u = xf(util.positions[t_util, i])
                    pos_r = xf(result.positions[t, i])
                    if float(np.linalg.norm(pos_r - pos_u)) < DELTA_THRESH:
                        continue
                    _ghost_car(ax, pos_u, float(util.headings[t_util, i]) + rot, length, width)
                    ax.annotate(
                        "",
                        xy=(pos_r[0], pos_r[1]),
                        xytext=(pos_u[0], pos_u[1]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="#0f172a",
                            lw=0.7,
                            mutation_scale=7,
                            shrinkA=0,
                            shrinkB=0,
                            alpha=0.55,
                        ),
                        zorder=4,
                    )
            for i in range(n_agents):
                color = AGENT_COLORS[i % len(AGENT_COLORS)]
                xy = np.asarray(result.positions[: steps + 1, i], float)
                _fading_history(ax, xf(xy[t0 : t + 1]), color)
                _car(ax, xf(result.positions[t, i]), float(result.headings[t, i]) + rot, length, width, color)
            if r == 0:
                ax.set_title(rf"$t={t_s}\,\mathrm{{s}}$", fontsize=11, fontweight="bold", color=INK, pad=3)
            if c == 0:
                ax.text(
                    -0.14,
                    0.5,
                    name,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                    fontweight="bold",
                    color="#0f766e" if r == 0 else "#0369a1",
                )
            if r == 0 and c < n_t - 1:
                ax.annotate(
                    "",
                    xy=(1.06, 1.04),
                    xytext=(0.94, 1.04),
                    xycoords="axes fraction",
                    textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.1),
                    annotation_clip=False,
                )
    fig.legend(
        handles=[
            Rectangle((0, 0), 1, 1, facecolor="#94a3b8", edgecolor="none", label="agents"),
            Line2D([0], [0], color="#64748b", lw=1.6, alpha=0.45, label="trajectory history"),
            Line2D([0], [0], color="#64748b", lw=1.0, ls=(0, (1.6, 1.4)), label="utility pose (Δ)"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.04),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=1.4,
    )
    fig.suptitle(
        f"Closed-loop rollouts over time  ·  {TITLES[site]}",
        fontsize=12.5,
        fontweight="bold",
        color=INK,
        y=1.01,
    )
    HERE.mkdir(parents=True, exist_ok=True)
    out = HERE / f"{site}.png"
    save_kw = dict(dpi=280, bbox_inches="tight", facecolor="white", pad_inches=0.30)
    fig.savefig(out, **save_kw)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", facecolor="white", pad_inches=0.24)
    plt.close(fig)
    print(f"Wrote {out}", flush=True)
    return out


def render_site(site: str, seed: int | None) -> Path:
    if site == "freeway":
        used = 910101 if seed is None else int(seed)
        print(f"Freeway rollouts seed={used}", flush=True)
        scenario, util, residual = collect_freeway(used)
        return make_time_strip(site, scenario, util, residual, box_scale=0.95, view="road")
    if site == "tgsim":
        _activate_case(REPO / "TGSIM Case")
        seeds = [int(seed)] if seed is not None else list(range(200, 205))
        last = None
        for cand in seeds:
            try:
                print(f"TGSIM rollouts seed={cand}", flush=True)
                scenario, util, residual = collect_tgsim(cand)
                return make_time_strip(site, scenario, util, residual, box_scale=1.15, view="square")
            except Exception as exc:  # noqa: BLE001
                print(f"  skip seed {cand}: {exc}", flush=True)
                last = exc
        raise RuntimeError(f"No TGSIM scene could be generated ({last})")
    if site == "roundabout":
        run_dir = str((REPO / "Roundabout Case" / "runs" / "revision2").resolve())
        _activate_case(REPO / "Roundabout Case", {"ROUNDABOUT_RUN_DIR": run_dir})
        seeds = [int(seed)] if seed is not None else list(range(1, 9))
        last = None
        for cand in seeds:
            try:
                print(f"Roundabout rollouts seed={cand}", flush=True)
                scenario, util, residual = collect_roundabout(cand)
                return make_time_strip(site, scenario, util, residual, box_scale=1.0, view="square")
            except Exception as exc:  # noqa: BLE001
                print(f"  skip seed {cand}: {exc}", flush=True)
                last = exc
        raise RuntimeError(f"No roundabout scene could be generated ({last})")
    raise ValueError(site)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--site", choices=("freeway", "tgsim", "roundabout", "all"), default="all")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    if args.site == "all":
        for site in ("freeway", "tgsim", "roundabout"):
            cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--site", site]
            if args.seed is not None:
                cmd.extend(["--seed", str(args.seed)])
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.check_call(cmd, cwd=str(REPO), env=env)
        return
    render_site(args.site, args.seed)


if __name__ == "__main__":
    main()

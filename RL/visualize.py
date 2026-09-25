#!/usr/bin/env python
"""Visualize utility-only vs residual-modulated traffic rollouts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D

import RL._paths  # noqa: F401
from RL.calibration_io import DEFAULT_CALIBRATION_PATH, load_base_params
from RL.corridor import (
    DEFAULT_LANE_KF,
    DEFAULT_RUN_ID,
    oriented_box_corners,
    load_corridor,
)
from RL.traffic_env import EnvConfig, MultiAgentTrafficEnv
from RL.calibration_io import RESIDUAL_PARAM_KEYS

try:
    import torch
    from RL.train_ppo import TorchResidualPolicy, action_space_from_blob, residual_mode_from_blob
except ImportError:
    torch = None
    TorchResidualPolicy = None


AGENT_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#003f5c",
    "#ffa600",
]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "figures" / "simulation"


def _draw_corridor(ax, corridor) -> None:
    ax.plot(corridor.center[:, 0], corridor.center[:, 1], "C0-", lw=1.5, alpha=0.8, label="center")
    ax.plot(corridor.lower[:, 0], corridor.lower[:, 1], "C3-", lw=2, label="lower")
    ax.plot(corridor.upper[:, 0], corridor.upper[:, 1], "C2-", lw=2, label="upper")
    ax.fill(
        list(corridor.lower[:, 0]) + list(corridor.upper[::-1, 0]),
        list(corridor.lower[:, 1]) + list(corridor.upper[::-1, 1]),
        color="C0",
        alpha=0.10,
        zorder=0,
    )


def _draw_vehicle(ax, pos, heading, color, length: float, width: float, zorder: int = 5):
    corners = oriented_box_corners(pos, heading, length=length, width=width)
    poly = np.vstack([corners, corners[0]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=0.85, edgecolor="k", lw=0.8, zorder=zorder)
    # nose mark
    nose = 0.5 * (corners[0] + corners[1])
    ax.plot([pos[0], nose[0]], [pos[1], nose[1]], color="k", lw=1.0, zorder=zorder + 1)


def load_policy(checkpoint: Path) -> Any | None:
    if torch is None or not checkpoint.exists():
        return None
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(checkpoint, map_location="cpu")
    from Baselines.residual_marl import load_residual_policy
    return load_residual_policy(checkpoint, int(payload["obs_dim"]), allow_legacy=True)


def record_rollout(
    env: MultiAgentTrafficEnv,
    policy=None,
    explore_std: float = 0.0,
) -> dict[str, Any]:
    """Run one episode and record positions, velocities, headings, and residuals."""
    if policy is not None:
        env.config.residual_mode = policy.residual_mode
    obs_list = env.reset()
    n_agents = len(env.agents)
    positions: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    velocities: list[list[np.ndarray]] = [[] for _ in range(n_agents)]
    headings: list[list[float]] = [[] for _ in range(n_agents)]
    residuals: list[list[dict[str, float]]] = [[] for _ in range(n_agents)]
    controls: list[list[dict[str, float]]] = [[] for _ in range(n_agents)]
    destinations = [a.dest.copy() for a in env.agents]
    start_positions = [a.pos.copy() for a in env.agents]
    length = float(env.config.sim_config.get("vehicle_length", 4.5))
    width = float(env.config.sim_config.get("vehicle_width", 1.8))

    for i in range(n_agents):
        positions[i].append(env.agents[i].pos.copy())
        velocities[i].append(env.agents[i].vel.copy())
        headings[i].append(float(env.agents[i].heading))
        residuals[i].append({})
        controls[i].append({"accel": 0.0, "steering": 0.0})

    done = False
    while not done:
        residual_actions = None
        if policy is not None:
            residual_actions = []
            for obs in obs_list:
                action, _ = policy.act(obs, explore_std=explore_std)
                residual_actions.append(action)

        obs_list, _, done, _ = env.step(residual_actions)
        for i in range(n_agents):
            positions[i].append(env.agents[i].pos.copy())
            velocities[i].append(env.agents[i].vel.copy())
            headings[i].append(float(env.agents[i].heading))
            controls[i].append(dict(env.agents[i].prev_control))
            if residual_actions is not None:
                action = residual_actions[i]
                if isinstance(action, dict):
                    residuals[i].append(dict(action))
                else:
                    residuals[i].append(
                        {f"c{j}": float(v) for j, v in enumerate(np.asarray(action).ravel())}
                    )
            else:
                residuals[i].append({})

    return {
        "positions": positions,
        "velocities": velocities,
        "headings": headings,
        "residuals": residuals,
        "controls": controls,
        "destinations": destinations,
        "starts": start_positions,
        "metric": env.rollout_metric(),
        "collisions": env.collision_count,
        "steps": env.step_count,
        "run_id": env.config.run_id,
        "lane_kf": env.config.lane_kf,
        "vehicle_length": length,
        "vehicle_width": width,
        "road_y": (
            env.config.sim_config["road_y_min"],
            env.config.sim_config["road_y_max"],
        ),
        "highway_length": env.config.highway_length,
    }


def _stack_positions(positions: list[list[np.ndarray]]) -> list[np.ndarray]:
    return [np.array(p) for p in positions]


def plot_trajectories(
    baseline: dict[str, Any],
    residual: dict[str, Any] | None,
    output_path: Path,
) -> None:
    """Side-by-side 2D trajectory plots on the measured corridor."""
    n_panels = 2 if residual is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(8 * n_panels, 7), constrained_layout=True)
    if n_panels == 1:
        axes = [axes]

    rolls = [("Utility-only baseline", baseline)]
    if residual is not None:
        rolls.append(("Residual policy", residual))

    for ax, (title, roll) in zip(axes, rolls):
        corridor = load_corridor(int(roll.get("run_id", DEFAULT_RUN_ID)), int(roll.get("lane_kf", DEFAULT_LANE_KF)))
        _draw_corridor(ax, corridor)
        length = float(roll.get("vehicle_length", 4.5))
        width = float(roll.get("vehicle_width", 1.8))
        pos_list = _stack_positions(roll["positions"])
        headings = roll.get("headings")
        for i, traj in enumerate(pos_list):
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            ax.plot(traj[:, 0], traj[:, 1], "-", color=color, lw=1.5, alpha=0.75, label=f"Agent {i}")
            h0 = headings[i][0] if headings is not None else 0.0
            h1 = headings[i][-1] if headings is not None else 0.0
            _draw_vehicle(ax, traj[0], h0, color, length, width, zorder=6)
            _draw_vehicle(ax, traj[-1], h1, color, length, width, zorder=6)
            dest = roll["destinations"][i]
            ax.scatter(dest[0], dest[1], s=90, c=color, marker="*", edgecolors="k", zorder=7)

        ax.set_title(
            f"{title} | run={roll.get('run_id')} lane={roll.get('lane_kf')}\n"
            f"metric={roll['metric']:.2f}, collisions={roll['collisions']}, steps={roll['steps']}",
            fontsize=11,
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_speed_profiles(baseline: dict[str, Any], residual: dict[str, Any] | None, output_path: Path) -> None:
    """Speed vs time for each agent."""
    n_agents = len(baseline["positions"])
    fig, axes = plt.subplots(n_agents, 1, figsize=(10, 3 * n_agents), sharex=True, constrained_layout=True)
    if n_agents == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        b_vel = np.array(baseline["velocities"][i])
        b_speed = np.linalg.norm(b_vel, axis=1)
        t = np.arange(len(b_speed)) * 0.5
        ax.plot(t, b_speed, "-", color=AGENT_COLORS[i], lw=2, label="Baseline")
        if residual is not None:
            r_vel = np.array(residual["velocities"][i])
            r_speed = np.linalg.norm(r_vel, axis=1)
            ax.plot(t, r_speed, "--", color=AGENT_COLORS[i], lw=2, alpha=0.8, label="Residual")
        ax.set_ylabel(f"Agent {i} speed (m/s)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Agent speed profiles", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_control_profiles(baseline: dict[str, Any], residual: dict[str, Any] | None, output_path: Path) -> None:
    """Acceleration and steering vs time for each agent."""
    n_agents = len(baseline["positions"])
    fig, axes = plt.subplots(n_agents, 2, figsize=(12, 2.5 * n_agents), sharex=True, constrained_layout=True)
    if n_agents == 1:
        axes = np.array([axes])

    dt = 0.5
    for i in range(n_agents):
        t = np.arange(len(baseline["controls"][i])) * dt
        b_ctrl = baseline["controls"][i]
        b_accel = [c.get("accel", 0.0) for c in b_ctrl]
        b_steer = [c.get("steering", 0.0) for c in b_ctrl]
        axes[i, 0].plot(t, b_accel, "-", color=AGENT_COLORS[i], lw=2, label="Baseline")
        axes[i, 1].plot(t, b_steer, "-", color=AGENT_COLORS[i], lw=2, label="Baseline")
        if residual is not None and "controls" in residual:
            r_ctrl = residual["controls"][i]
            r_accel = [c.get("accel", 0.0) for c in r_ctrl]
            r_steer = [c.get("steering", 0.0) for c in r_ctrl]
            axes[i, 0].plot(t, r_accel, "--", color=AGENT_COLORS[i], lw=2, alpha=0.8, label="Residual")
            axes[i, 1].plot(t, r_steer, "--", color=AGENT_COLORS[i], lw=2, alpha=0.8, label="Residual")
        axes[i, 0].set_ylabel(f"Agent {i}\naccel (m/s2)")
        axes[i, 1].set_ylabel(f"Agent {i}\nsteer (rad)")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 1].grid(True, alpha=0.3)
        if i == 0:
            axes[i, 0].legend(fontsize=8)
            axes[i, 1].legend(fontsize=8)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Time (s)")
    fig.suptitle("Utility-selected acceleration and steering", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def residual_series_matrix(series):
    """Return the actual emitted coordinates instead of silently plotting zeros."""
    present = {key for delta in series for key in delta if isinstance(delta, dict)}
    if present and all(key.startswith("c") and key[1:].isdigit() for key in present):
        keys = sorted(present, key=lambda key: int(key[1:]))
    else:
        keys = [key for key in RESIDUAL_PARAM_KEYS if key in present]
        keys += sorted(present - set(keys))
    matrix = np.array([[abs(float(delta.get(key, 0.0))) for delta in series[1:]] for key in keys])
    return keys, matrix


def plot_residual_heatmap(residual: dict[str, Any], output_path: Path) -> None:
    """Heatmap of |dTheta| components over time per agent."""
    n_agents = len(residual["residuals"])
    fig, axes = plt.subplots(n_agents, 1, figsize=(11, 2.8 * n_agents), constrained_layout=True)
    if n_agents == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        series = residual["residuals"][i]
        keys, mat = residual_series_matrix(series)
        if not keys or mat.size == 0:
            ax.set_visible(False)
            continue
        im = ax.imshow(mat, aspect="auto", cmap="coolwarm", origin="lower")
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels(keys, fontsize=9)
        ax.set_xlabel("Step")
        ax.set_title(f"Agent {i}: emitted residuals")
        fig.colorbar(im, ax=ax, label="|residual|")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_animation(roll: dict[str, Any], output_path: Path, title: str, fps: int = 4) -> None:
    """Animated 2D rollout (GIF) with corridor + rectangular cars."""
    from matplotlib.patches import Polygon

    pos_list = _stack_positions(roll["positions"])
    headings = roll.get("headings")
    n_frames = max(len(p) for p in pos_list)
    corridor = load_corridor(int(roll.get("run_id", DEFAULT_RUN_ID)), int(roll.get("lane_kf", DEFAULT_LANE_KF)))
    length = float(roll.get("vehicle_length", 4.5))
    width = float(roll.get("vehicle_width", 1.8))

    all_x = np.concatenate([corridor.lower[:, 0], corridor.upper[:, 0], *[p[:, 0] for p in pos_list]])
    all_y = np.concatenate([corridor.lower[:, 1], corridor.upper[:, 1], *[p[:, 1] for p in pos_list]])
    pad = 5.0
    xlim = (float(all_x.min()) - pad, float(all_x.max()) + pad)
    ylim = (float(all_y.min()) - pad, float(all_y.max()) + pad)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    _draw_corridor(ax, corridor)

    lines = []
    car_patches = []
    for i, dest in enumerate(roll["destinations"]):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        (ln,) = ax.plot([], [], "-", color=color, lw=1.5, alpha=0.7)
        patch = Polygon([[0, 0], [0, 0], [0, 0], [0, 0]], closed=True, facecolor=color, edgecolor="k", alpha=0.9, zorder=5)
        ax.add_patch(patch)
        ax.plot([dest[0]], [dest[1]], "*", color=color, ms=12, markeredgecolor="k")
        lines.append(ln)
        car_patches.append(patch)

    legend_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=AGENT_COLORS[i], label=f"Agent {i}", markersize=8)
        for i in range(len(pos_list))
    ]
    ax.legend(handles=legend_handles, loc="best")

    def update(frame: int):
        artists = []
        for i, traj in enumerate(pos_list):
            sub = traj[: frame + 1]
            lines[i].set_data(sub[:, 0], sub[:, 1])
            heading = headings[i][min(frame, len(headings[i]) - 1)] if headings is not None else 0.0
            corners = oriented_box_corners(sub[-1], heading, length=length, width=width)
            car_patches[i].set_xy(corners)
            artists.extend([lines[i], car_patches[i]])
        ax.set_title(f"{title} — step {frame}/{n_frames - 1}")
        return artists

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize traffic rollouts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--num-agents", type=int, default=10)
    parser.add_argument("--run-id", type=int, default=DEFAULT_RUN_ID)
    parser.add_argument("--lane-kf", type=int, default=DEFAULT_LANE_KF)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION_PATH)
    parser.add_argument("--prefer-params", choices=("working", "robust", "best"), default="working")
    parser.add_argument("--checkpoint", type=Path, default=Path("RL/checkpoints/residual_policy.pt"))
    parser.add_argument("--no-gif", action="store_true", help="Skip GIF animation export")
    parser.add_argument("--baseline-only", action="store_true", help="Skip residual checkpoint even if present")
    args = parser.parse_args()

    base_params = None
    if args.calibration.exists():
        base_params = load_base_params(args.calibration, prefer=args.prefer_params)
        print(f"Theta_base = {args.prefer_params} from {args.calibration}")

    env_cfg = EnvConfig(
        max_steps=args.max_steps,
        num_agents=args.num_agents,
        base_params=base_params,
        run_id=args.run_id,
        lane_kf=args.lane_kf,
    )
    print(f"Corridor: run_id={args.run_id}, lane_kf={args.lane_kf}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Recording baseline rollout...")
    baseline_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
    baseline = record_rollout(baseline_env, policy=None)

    residual = None
    if not args.baseline_only:
        policy = load_policy(args.checkpoint)
        if policy is not None:
            print("Recording PyTorch residual rollout...")
            residual_env = MultiAgentTrafficEnv(env_cfg, seed=args.seed)
            residual = record_rollout(residual_env, policy=policy, explore_std=0.0)
        else:
            print("No checkpoint found — plotting baseline only.")

    plot_trajectories(baseline, residual, OUTPUT_DIR / "trajectories_compare.png")
    print(f"Saved {OUTPUT_DIR / 'trajectories_compare.png'}")

    plot_speed_profiles(baseline, residual, OUTPUT_DIR / "speed_profiles.png")
    print(f"Saved {OUTPUT_DIR / 'speed_profiles.png'}")

    plot_control_profiles(baseline, residual, OUTPUT_DIR / "control_profiles.png")
    print(f"Saved {OUTPUT_DIR / 'control_profiles.png'}")

    if residual is not None:
        plot_residual_heatmap(residual, OUTPUT_DIR / "residual_heatmap.png")
        print(f"Saved {OUTPUT_DIR / 'residual_heatmap.png'}")

    if not args.no_gif:
        print("Rendering baseline animation...")
        save_animation(baseline, OUTPUT_DIR / "baseline_rollout.gif", "Utility-only baseline")
        print(f"Saved {OUTPUT_DIR / 'baseline_rollout.gif'}")
        if residual is not None:
            print("Rendering residual animation...")
            save_animation(residual, OUTPUT_DIR / "residual_rollout.gif", "Residual policy")
            print(f"Saved {OUTPUT_DIR / 'residual_rollout.gif'}")

    print("\nDone.")


if __name__ == "__main__":
    main()

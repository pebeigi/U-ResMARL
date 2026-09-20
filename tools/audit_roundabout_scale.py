"""Plot recorded cars against the unmodified Jounieh curb and report scale checks.

Run from the repository root: python tools/audit_roundabout_scale.py
The stored metres-per-pixel value is retained, not independently calibrated.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as Patch, Rectangle
import numpy as np
import pandas as pd
import shapely

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Calibration.calibrate_utility_from_data import load_site_roadway
from RL.corridor import oriented_box_corners


def main():
    folder = ROOT / "data/Lebanon_Jounieh"
    frame = pd.read_csv(folder / "prepared/trajectories_calibration.csv").sort_values(["id", "time"])
    road = load_site_roadway(folder / "Jounieh_Road_Boundaries.csv")
    shapely.prepare(road)
    inside = shapely.covers(road, shapely.points(frame[["xloc_kf", "yloc_kf"]].to_numpy()))
    delta = frame.groupby("id")[["xloc_kf", "yloc_kf", "time"]].diff()
    speed = np.hypot(delta.xloc_kf, delta.yloc_kf) / delta.time
    heading = np.arctan2(delta.yloc_kf, delta.xloc_kf)
    mask = (speed > .8) & (delta.time < .2) & frame.keep_ego
    c, s = np.abs(np.cos(heading[mask])), np.abs(np.sin(heading[mask]))
    design = np.concatenate([np.column_stack([c, s]), np.column_stack([s, c])])
    # Treat width_smoothed as the image x extent and length_smoothed as y.
    # This is a diagnostic hypothesis, not verified oriented-box metadata.
    dims = frame.loc[mask, ["width_smoothed", "length_smoothed"]].to_numpy()
    target = np.r_[dims[:, 0], dims[:, 1]]
    fitted, *_ = np.linalg.lstsq(design, target, rcond=None)

    time = 100.0
    snapshot = frame[np.isclose(frame.time, time) & frame.keep_ego & (frame.speed_kf > .8)].copy()
    selected = []
    for row in snapshot.sort_values("id").itertuples():
        xy = np.array([row.xloc_kf, row.yloc_kf])
        if not road.covers(shapely.Point(xy)) or any(np.linalg.norm(xy - p[1]) < 4 for p in selected):
            continue
        track = frame[frame.id == row.id]
        past = track[track.time <= time - .5].iloc[-1]
        vel = (xy - [past.xloc_kf, past.yloc_kf]) / (time - past.time)
        selected.append((row, xy, float(np.arctan2(vel[1], vel[0]))))
        if len(selected) == 6:
            break
    if len(selected) < 3:
        raise ValueError("Insufficient recorded cars for scale comparison")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.8), sharex=True, sharey=True)
    titles = ["Current simulator: 1.60 × 0.64 m", "Calibration footprint: 4.50 × 1.80 m",
              "Recorded bounding boxes (axis-aligned)"]
    colors = plt.get_cmap("tab10").colors
    for panel, (ax, title) in enumerate(zip(axes, titles)):
        ax.add_patch(Patch(np.asarray(road.exterior.coords), facecolor="#dfe6eb", edgecolor="#273e50", lw=1.4))
        for hole in road.interiors:
            ax.add_patch(Patch(np.asarray(hole.coords), facecolor="#fffaf0", edgecolor="#273e50", lw=1.2))
        for i, (row, xy, yaw) in enumerate(selected):
            color = colors[i]
            trace = frame[(frame.id == row.id) & (frame.time.between(time - 2, time + 2))]
            ax.plot(trace.xloc_kf, trace.yloc_kf, color=color, lw=1.1, alpha=.6)
            if panel < 2:
                length, width = ((1.6, .64), (4.5, 1.8))[panel]
                box = oriented_box_corners(xy, yaw, length, width)
                ax.add_patch(Patch(box, facecolor=color, edgecolor="black", lw=.8, alpha=.85))
            else:
                ax.add_patch(Rectangle(xy - [row.width_smoothed / 2, row.length_smoothed / 2],
                                      row.width_smoothed, row.length_smoothed,
                                      facecolor=color, edgecolor="black", lw=.8, alpha=.85))
            ax.annotate("", xy=xy + 2 * np.array([np.cos(yaw), np.sin(yaw)]), xytext=xy,
                        arrowprops=dict(arrowstyle="->", color="black", lw=1))
            ax.text(xy[0] + .7, xy[1] - 1.2, str(row.id), fontsize=7)
        ax.set_title(title, fontsize=11)
        ax.set_aspect("equal")
        ax.set_xlim(-1, 57)
        ax.set_ylim(33, -1)  # Source origin is image top-left; y points down.
        ax.set_xlabel("x (m, using supplied scale)")
        ax.grid(alpha=.12)
    axes[0].set_ylabel("y (m, using supplied scale; image-down axis)")
    fig.suptitle("Jounieh scale check — identical recorded cars and unmodified curb", fontsize=15)
    fig.text(.5, .12, "Source: prepared Jounieh trajectories at t = 100 s; lines show t = 98–102 s; arrows show observed direction.\n"
             "Supplied scale: 0.04354578 m/pixel. Panel 3 assumes width_smoothed = x extent, length_smoothed = y extent.\n"
             "Visual consistency does not verify the absolute metre scale; a measured ground distance is still needed.",
             ha="center", va="top", fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.23, left=.05, right=.99, wspace=.08)
    output = ROOT / "Roundabout Case/figures"
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "scale_audit.png", dpi=180)
    plt.close(fig)
    report = dict(supplied_metres_per_pixel=.04354578, absolute_scale_verified=False,
                  curb_bounds=list(road.bounds), trajectory_bounds={
                      key: [float(frame[key].min()), float(frame[key].max())] for key in ("xloc_kf", "yloc_kf")},
                  recorded_centres_inside_curb_fraction=float(inside.mean()),
                  recorded_box_quantiles=frame[["length_smoothed", "width_smoothed"]].quantile([.05, .5, .95]).to_dict(),
                  inferred_oriented_length_width=fitted.tolist(),
                  box_fit_rmse=float(np.sqrt(np.mean((design @ fitted - target) ** 2))),
                  box_fit_note="Hypothesis: width_smoothed is image x extent and length_smoothed is y extent; not physical ground truth.",
                  recorded_speed_median=float(frame.speed_kf.median()),
                  coordinate_difference_speed_median=float(speed.median()),
                  snapshot_time_s=time, plotted_vehicle_ids=[int(p[0].id) for p in selected],
                  note="A scale correction must transform curb, trajectories, dimensions, speeds and accelerations together, then refit the utility prior.")
    (output / "scale_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print(output / "scale_audit.png", flush=True)


if __name__ == "__main__":
    main()

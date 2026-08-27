#!/usr/bin/env python
"""Surface-based sensitivity visualizations for utility components.

Aligned with the current NR-MAP utility formulation:
  - Direction: corridor tangent alignment
  - Speed: candidate speed vs desired speed (symmetric rho)
  - Proximity: Frenet / effective distance vs horizon
  - Collision / path: unchanged shapes

With three varied parameters plus a utility value, a single ordinary 3D surface
is not possible because that would require four dimensions. The response-surface
standard is to plot two parameters on x/y, utility on z, and show the third
parameter as low/mid/high slices.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import Sensitivity._paths  # noqa: F401
from Sensitivity._paths import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "figures" / "sensitivity" / "3d"
APPENDIX_DIR = REPO_ROOT / "Paper Draft" / "Appendix"

# Robust highway working point used in the main experiments (theta_rob).
COLLISION_W_C = 300.0
VEHICLE_LENGTH_M = 4.5
VEHICLE_WIDTH_M = 1.8
SIGMA_PARALLEL_M = 0.89
SIGMA_PERP_ROB_M = 0.59


def directional_utility(s_theta: np.ndarray, alignment_cos: np.ndarray) -> np.ndarray:
    """Corridor-frame direction utility: U_dir = S_theta * (m_hat · t_hat)."""
    return s_theta * alignment_cos


def speed_utility(s_v: np.ndarray, v_cand: np.ndarray, v_des: np.ndarray, xi_i: np.ndarray) -> np.ndarray:
    """Symmetric candidate-speed vs desired-speed utility."""
    v_cand = np.maximum(v_cand, 0.0)
    v_des = np.maximum(v_des, 1e-6)
    rho = np.minimum(v_cand, v_des) / np.maximum(v_cand, v_des)
    rho = np.maximum(rho, 1e-6)
    return s_v * (rho / (1 + rho ** ((xi_i - 1) / 2)))


def proximity_utility(
    s_d: np.ndarray,
    gamma: np.ndarray,
    d_eff: np.ndarray,
    h_p: np.ndarray,
) -> np.ndarray:
    """Proximity / progress utility over effective Frenet distance."""
    h_p = np.maximum(h_p, 1e-6)
    return s_d / (1 + (d_eff / h_p) ** gamma)


def collision_utility(
    w_c: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    sigma_parallel: np.ndarray,
    sigma_perp: np.ndarray,
    length: float = VEHICLE_LENGTH_M,
    width: float = VEHICLE_WIDTH_M,
) -> np.ndarray:
    """Single-neighbor OBB surface-gap penalty used by utility_model.collision_penalty.

    Presence is 1 on overlapping L×W footprints and decays as an anisotropic
    Gaussian of the surface gaps (not a centre-to-centre PDF).
    """
    gap_parallel = np.maximum(0.0, np.abs(dx) - length)
    gap_perp = np.maximum(0.0, np.abs(dy) - width)
    sig_par = np.maximum(sigma_parallel, 1e-6)
    sig_lat = np.maximum(sigma_perp, 1e-6)
    presence = np.exp(-0.5 * ((gap_parallel / sig_par) ** 2 + (gap_perp / sig_lat) ** 2))
    return -w_c * presence


def path_adherence_utility(
    w_ell: np.ndarray,
    beta: np.ndarray,
    ell_i: np.ndarray,
) -> np.ndarray:
    """Path / boundary adherence as a negative utility penalty."""
    return -w_ell * (1 - np.exp(-beta * ell_i**2))


def plot_response_surface_slices(
    x_values: np.ndarray,
    y_values: np.ndarray,
    slice_values: list[float],
    utility_fn,
    labels: tuple[str, str, str],
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    show: bool = False,
) -> None:
    """Plot utility response surfaces for low/mid/high values of a third parameter."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_grid, y_grid = np.meshgrid(x_values, y_values, indexing="xy")
    utilities = [utility_fn(x_grid, y_grid, slice_value) for slice_value in slice_values]
    vmin = min(float(np.nanmin(utility)) for utility in utilities)
    vmax = max(float(np.nanmax(utility)) for utility in utilities)
    if abs(vmax - vmin) < 1e-12:
        vmax = vmin + 1e-6

    n_panels = len(slice_values)
    fig = plt.figure(figsize=(5.8 * n_panels + 1.2, 5.8))

    for index, (slice_value, utility_grid) in enumerate(zip(slice_values, utilities), start=1):
        ax = fig.add_subplot(1, n_panels, index, projection="3d")
        ax.plot_surface(
            x_grid,
            y_grid,
            utility_grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0,
            antialiased=True,
            alpha=0.96,
        )
        ax.contour(
            x_grid,
            y_grid,
            utility_grid,
            zdir="z",
            offset=vmin,
            cmap=cmap,
            levels=10,
            linewidths=0.7,
        )
        if labels[2] and labels[2] != "(n/a)":
            ax.set_title(f"{labels[2]} = {slice_value:g}", fontsize=12)
        else:
            ax.set_title("Directional utility", fontsize=12)
        ax.set_xlabel(labels[0])
        ax.set_ylabel(labels[1])
        ax.set_zlabel("Utility")
        ax.set_zlim(vmin, vmax)
        ax.view_init(elev=28, azim=-135)

    # Dedicated right strip for the colorbar so it never overlays the panels.
    fig.subplots_adjust(left=0.04, right=0.86, bottom=0.08, top=0.86, wspace=0.18)
    cax = fig.add_axes([0.90, 0.18, 0.018, 0.58])
    mappable = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    mappable.set_array([])
    colorbar = fig.colorbar(mappable, cax=cax)
    colorbar.set_label("Utility value")

    fig.suptitle(title, fontsize=16)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_directional_3d(show: bool = False) -> None:
    s_theta = np.linspace(0.2, 2.0, 80)
    alignment = np.linspace(-1.0, 1.0, 80)
    # Constant slices keep a consistent panel format with the other plots.
    dummy_slices = [0.0]
    plot_response_surface_slices(
        s_theta,
        alignment,
        dummy_slices,
        utility_fn=lambda s_grid, cos_grid, _unused: directional_utility(s_grid, cos_grid),
        labels=(r"$S_\theta$", r"$\hat{m}\cdot\hat{t}$", r"(n/a)"),
        title="Directional Utility Response Surfaces (Corridor Tangent Alignment)",
        output_path=OUTPUT_DIR / "directional_utility_surface_slices.png",
        cmap="viridis",
        show=show,
    )


def plot_speed_3d(show: bool = False) -> None:
    v_cand = np.linspace(0.0, 22.0, 80)
    v_des = np.linspace(2.0, 22.0, 80)
    xi_slices = [1.5, 2.5, 3.5]
    plot_response_surface_slices(
        v_cand,
        v_des,
        xi_slices,
        utility_fn=lambda v_cand_grid, v_des_grid, xi: speed_utility(
            1.0, v_cand_grid, v_des_grid, xi
        ),
        labels=(r"$v_{\mathrm{cand}}$", r"$v_{\mathrm{des}}$", r"$\xi_i$"),
        title="Speed Utility Response Surfaces (Candidate vs Desired)",
        output_path=OUTPUT_DIR / "speed_utility_surface_slices.png",
        cmap="viridis",
        show=show,
    )


def plot_proximity_3d(show: bool = False) -> None:
    gamma = np.linspace(0.5, 3.0, 80)
    d_eff = np.linspace(1.0, 150.0, 80)
    h_p_slices = [20.0, 60.0, 100.0]
    plot_response_surface_slices(
        gamma,
        d_eff,
        h_p_slices,
        utility_fn=lambda gamma_grid, d_grid, h_p: proximity_utility(
            1.0,
            gamma_grid,
            d_grid,
            h_p,
        ),
        labels=(r"$\gamma$", r"$d_{\mathrm{eff}}=w_x|\Delta s|+w_y|\Delta n|$", r"$H_p$"),
        title="Proximity Utility Response Surfaces (Frenet Effective Distance)",
        output_path=OUTPUT_DIR / "proximity_utility_surface_slices.png",
        cmap="plasma",
        show=show,
    )


def plot_collision_3d(show: bool = False) -> None:
    dx = np.linspace(0.0, 12.0, 80)
    dy = np.linspace(0.0, 6.0, 80)
    # Hold theta_rob (w_c, sigma_parallel, L×W); slice the lateral kernel width.
    sigma_perp_slices = [0.30, SIGMA_PERP_ROB_M, 1.20]
    output_path = OUTPUT_DIR / "collision_utility_surface_slices.png"
    plot_response_surface_slices(
        dx,
        dy,
        sigma_perp_slices,
        utility_fn=lambda dx_grid, dy_grid, sigma_perp: collision_utility(
            COLLISION_W_C,
            dx_grid,
            dy_grid,
            SIGMA_PARALLEL_M,
            sigma_perp,
        ),
        labels=(r"$|\Delta x|$ (m)", r"$|\Delta y|$ (m)", r"$\sigma_{\perp}$"),
        title=(
            r"Collision Utility Response Surfaces "
            r"(OBB surface-gap kernel at $\theta_{\mathrm{rob}}$)"
        ),
        output_path=output_path,
        cmap="magma",
        show=show,
    )
    appendix_path = APPENDIX_DIR / "collision_utility_surface_slices.png"
    appendix_path.parent.mkdir(parents=True, exist_ok=True)
    appendix_path.write_bytes(output_path.read_bytes())


def plot_path_3d(show: bool = False) -> None:
    w_ell = np.linspace(5.0, 50.0, 80)
    beta = np.linspace(0.1, 5.0, 80)
    ell_slices = [0.5, 2.5, 5.0]
    plot_response_surface_slices(
        w_ell,
        beta,
        ell_slices,
        utility_fn=lambda w_grid, beta_grid, ell_i: path_adherence_utility(
            w_grid,
            beta_grid,
            ell_i,
        ),
        labels=(r"$w_{\ell}$", r"$\beta$", r"$\ell_i$"),
        title="Path-Adherence Utility Response Surfaces",
        output_path=OUTPUT_DIR / "path_adherence_utility_surface_slices.png",
        cmap="cividis",
        show=show,
    )


def main(show: bool = False) -> None:
    plot_directional_3d(show=show)
    plot_speed_3d(show=show)
    plot_proximity_3d(show=show)
    plot_collision_3d(show=show)
    plot_path_3d(show=show)
    print(f"Saved 3D sensitivity plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main(show=False)

"""
Visualization for PGDR experiments.

Generates all figures referenced in the proposal:
    1. Covariance calibration scatter plot (the key diagnostic)
    2. Bar chart comparing C1–C4 on velocity tracking error
    3. CMA-ES convergence curves
    4. Sensitivity ranking plots
    5. Covariance eigenvalue spectrum
    6. Per-parameter uncertainty vs error scatter (grouped by type)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for TACC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# Consistent style across all figures
COLORS = {
    "C1_uniform_dr": "#4C72B0",
    "C2_pure_sysid": "#DD8452",
    "C3_isotropic": "#55A868",
    "C4_pgdr_0.5": "#C44E52",
    "C4_pgdr_1.0": "#8172B3",
    "C4_pgdr_2.0": "#937860",
}

LABELS = {
    "C1_uniform_dr": "C1: Uniform DR",
    "C2_pure_sysid": "C2: Pure Sys-ID",
    "C3_isotropic": "C3: Isotropic",
    "C4_pgdr_0.5": r"C4: PGDR $\alpha$=0.5",
    "C4_pgdr_1.0": r"C4: PGDR $\alpha$=1.0",
    "C4_pgdr_2.0": r"C4: PGDR $\alpha$=2.0",
}

GROUP_COLORS = {
    "friction": "#E24A33",
    "mass": "#348ABD",
    "actuator": "#988ED5",
    "contact": "#FBC15E",
}


def setup_style():
    """Set up publication-quality plot style."""
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
    })


# ---------------------------------------------------------------------------
# 1. Covariance calibration plot
# ---------------------------------------------------------------------------

def plot_covariance_calibration(
    calibration_data: dict,
    save_path: str = "pgdr/results/covariance_calibration.png",
):
    """
    Scatter plot: diag(Σ)[i] vs (p*[i] - p_true[i])².

    This is the key diagnostic for PGDR. A positive correlation validates
    the core assumption that CMA-ES covariance tracks real uncertainty.

    Points are colored by parameter group.
    """
    setup_style()
    fig, ax = plt.subplots(figsize=(6, 5))

    per_param = calibration_data["per_param"]
    pearson = calibration_data["pearson_correlation"]
    spearman = calibration_data["spearman_correlation"]

    for entry in per_param:
        color = GROUP_COLORS.get(entry["group"], "#333333")
        ax.scatter(
            entry["uncertainty"],
            entry["sq_error"],
            c=color,
            s=30,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.3,
        )

    # Trend line
    x = np.array([e["uncertainty"] for e in per_param])
    y = np.array([e["sq_error"] for e in per_param])
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax.plot(x_line, p(x_line), "--", color="gray", alpha=0.5, linewidth=1)

    # Legend for groups
    for group, color in GROUP_COLORS.items():
        ax.scatter([], [], c=color, s=30, label=group.capitalize())
    ax.legend(loc="upper left", framealpha=0.8)

    ax.set_xlabel(r"$\mathrm{diag}(\Sigma)_i$ (CMA-ES uncertainty)")
    ax.set_ylabel(r"$(p^*_i - p^{\mathrm{true}}_i)^2$ (identification error)")
    ax.set_title(
        f"Covariance Calibration\n"
        f"Pearson r = {pearson:.3f}, Spearman ρ = {spearman:.3f}"
    )

    # Log scale if range is large
    if x.max() / max(x.min(), 1e-10) > 100:
        ax.set_xscale("log")
    if y.max() / max(y.min(), 1e-10) > 100:
        ax.set_yscale("log")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved covariance calibration plot to {save_path}")


# ---------------------------------------------------------------------------
# 2. Condition comparison bar chart
# ---------------------------------------------------------------------------

def plot_condition_comparison(
    eval_results: dict,
    save_path: str = "pgdr/results/condition_comparison.png",
):
    """
    Bar chart comparing all conditions on velocity tracking error.

    Shows nominal and perturbed performance side by side.
    """
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    conditions = [k for k in eval_results if k.startswith("C")]
    if not conditions:
        print("No condition results found for plotting.")
        plt.close(fig)
        return

    # Nominal performance
    ax = axes[0]
    names = [LABELS.get(c, c) for c in conditions]
    vals_nom = [eval_results[c].get("rms_vel_error_nominal", 0) for c in conditions]
    stds_nom = [eval_results[c].get("rms_vel_error_nominal_std", 0) for c in conditions]
    colors = [COLORS.get(c, "#666666") for c in conditions]

    bars = ax.bar(range(len(conditions)), vals_nom, yerr=stds_nom,
                  color=colors, edgecolor="white", linewidth=0.5,
                  capsize=3)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("RMS Velocity Error (m/s)")
    ax.set_title("Nominal Performance")

    # Perturbed performance
    ax = axes[1]
    vals_pert = [eval_results[c].get("rms_vel_error_perturbed", 0) for c in conditions]
    stds_pert = [eval_results[c].get("rms_vel_error_perturbed_std", 0) for c in conditions]

    bars = ax.bar(range(len(conditions)), vals_pert, yerr=stds_pert,
                  color=colors, edgecolor="white", linewidth=0.5,
                  capsize=3)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("RMS Velocity Error (m/s)")
    ax.set_title("Under Perturbation (+1kg payload)")

    fig.suptitle("Condition Comparison: Velocity Tracking", fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved condition comparison to {save_path}")


# ---------------------------------------------------------------------------
# 3. CMA-ES convergence curve
# ---------------------------------------------------------------------------

def plot_convergence(
    history: dict,
    save_path: str = "pgdr/results/convergence.png",
):
    """Plot CMA-ES optimization convergence."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    gens = history["generation"]
    best = history["best_loss"]
    mean = history["mean_loss"]
    sigma = history["sigma"]

    ax1.semilogy(gens, best, label="Best", color="#4C72B0", linewidth=1.5)
    ax1.semilogy(gens, mean, label="Mean", color="#DD8452", linewidth=1,
                 alpha=0.7)
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.set_title("CMA-ES Convergence")
    ax1.grid(True, alpha=0.3)

    ax2.plot(gens, sigma, color="#55A868", linewidth=1.5)
    ax2.set_ylabel(r"Step size $\sigma$")
    ax2.set_xlabel("Generation")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved convergence plot to {save_path}")


# ---------------------------------------------------------------------------
# 4. Sensitivity ranking
# ---------------------------------------------------------------------------

def plot_sensitivity_ranking(
    sensitivity_data: dict,
    top_k: int = 30,
    save_path: str = "pgdr/results/sensitivity_ranking.png",
):
    """Horizontal bar chart of parameter sensitivity scores."""
    setup_style()
    fig, ax = plt.subplots(figsize=(8, max(6, top_k * 0.25)))

    ranking = sensitivity_data["ranking"][:top_k]
    names = [r["param_name"] for r in reversed(ranking)]
    scores = [r["score"] for r in reversed(ranking)]
    groups = [r["group"] for r in reversed(ranking)]
    colors = [GROUP_COLORS.get(g, "#666666") for g in groups]

    ax.barh(range(len(names)), scores, color=colors, edgecolor="white",
            linewidth=0.3)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Sensitivity Score (trajectory divergence)")
    ax.set_title(f"Top {top_k} Most Sensitive Parameters")

    # Legend
    for group, color in GROUP_COLORS.items():
        ax.barh([], [], color=color, label=group.capitalize())
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved sensitivity ranking to {save_path}")


# ---------------------------------------------------------------------------
# 5. Covariance eigenvalue spectrum
# ---------------------------------------------------------------------------

def plot_eigenvalue_spectrum(
    eigenvalues: list[float],
    save_path: str = "pgdr/results/eigenvalue_spectrum.png",
):
    """Plot the eigenvalue spectrum of the CMA-ES covariance."""
    setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    eigvals = np.array(sorted(eigenvalues, reverse=True))
    idx = np.arange(1, len(eigvals) + 1)

    # Linear scale
    ax1.bar(idx, eigvals, color="#4C72B0", edgecolor="white", linewidth=0.3)
    ax1.set_xlabel("Eigenvalue index")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Covariance Spectrum")

    # Log scale
    ax2.semilogy(idx, eigvals, "o-", color="#4C72B0", markersize=3)
    ax2.set_xlabel("Eigenvalue index")
    ax2.set_ylabel("Eigenvalue (log)")
    ax2.set_title("Covariance Spectrum (log scale)")
    ax2.grid(True, alpha=0.3)

    # Cumulative variance explained
    cumvar = np.cumsum(eigvals) / np.sum(eigvals)
    ax_twin = ax2.twinx()
    ax_twin.plot(idx, cumvar, "--", color="#DD8452", linewidth=1)
    ax_twin.set_ylabel("Cumulative variance explained", color="#DD8452")
    ax_twin.tick_params(axis="y", labelcolor="#DD8452")

    fig.suptitle(
        f"Σ Eigenvalue Spectrum (d={len(eigvals)}, "
        f"condition number={eigvals[0]/max(eigvals[-1], 1e-10):.0f})",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved eigenvalue spectrum to {save_path}")


# ---------------------------------------------------------------------------
# 6. Per-parameter uncertainty comparison (PGDR vs isotropic)
# ---------------------------------------------------------------------------

def plot_pgdr_vs_isotropic(
    Sigma: np.ndarray,
    alpha: float,
    param_names: list[str],
    param_groups: list[str],
    save_path: str = "pgdr/results/pgdr_vs_isotropic.png",
):
    """
    Show how PGDR and isotropic DR differ in per-parameter std.

    This visualizes the core PGDR insight: PGDR allocates more noise
    to uncertain parameters and less to well-identified ones.
    """
    setup_style()
    d = len(param_names)

    pgdr_std = np.sqrt(np.diag(alpha * Sigma))
    total_var = np.trace(alpha * Sigma)
    iso_beta = np.sqrt(total_var / d)
    iso_std = np.full(d, iso_beta)

    # Sort by PGDR std descending
    order = np.argsort(-pgdr_std)
    pgdr_sorted = pgdr_std[order]
    iso_sorted = iso_std[order]
    names_sorted = [param_names[i] for i in order]
    groups_sorted = [param_groups[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, max(6, d * 0.2)))

    y = np.arange(d)
    bar_height = 0.35

    bars_pgdr = ax.barh(y + bar_height/2, pgdr_sorted, bar_height,
                         label="PGDR (anisotropic)", color="#8172B3",
                         edgecolor="white", linewidth=0.3)
    bars_iso = ax.barh(y - bar_height/2, iso_sorted, bar_height,
                        label="Isotropic (matched trace)", color="#55A868",
                        alpha=0.7, edgecolor="white", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels(names_sorted, fontsize=6)
    ax.set_xlabel("Per-parameter standard deviation (normalized)")
    ax.set_title(
        f"PGDR vs Isotropic DR (α={alpha}, total variance={total_var:.2f})"
    )
    ax.legend()
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved PGDR vs isotropic comparison to {save_path}")


# ---------------------------------------------------------------------------
# Master plotting function
# ---------------------------------------------------------------------------

def generate_all_plots(results_dir: str = "pgdr/results"):
    """Generate all plots from saved results."""
    results_dir = Path(results_dir)

    # 1. Covariance calibration
    cal_path = results_dir / "calibration.json"
    if cal_path.exists():
        cal = json.loads(cal_path.read_text())
        plot_covariance_calibration(cal, str(results_dir / "covariance_calibration.png"))

    # 2. Condition comparison
    eval_path = results_dir / "eval_results.json"
    if eval_path.exists():
        eval_results = json.loads(eval_path.read_text())
        plot_condition_comparison(eval_results,
                                  str(results_dir / "condition_comparison.png"))

    # 3. Convergence
    sysid_info_path = results_dir / "sysid_info.json"
    if sysid_info_path.exists():
        info = json.loads(sysid_info_path.read_text())
        if "history" in info:
            plot_convergence(info["history"],
                            str(results_dir / "convergence.png"))

    # 4. Sensitivity ranking
    sens_path = results_dir / "sensitivity_ranking.json"
    if sens_path.exists():
        sens = json.loads(sens_path.read_text())
        plot_sensitivity_ranking(sens, save_path=str(results_dir / "sensitivity_ranking.png"))

    # 5. Eigenvalue spectrum
    sigma_path = results_dir / "Sigma.npy"
    if sigma_path.exists():
        Sigma = np.load(str(sigma_path))
        eigvals = sorted(np.linalg.eigvalsh(Sigma).tolist(), reverse=True)
        plot_eigenvalue_spectrum(eigvals,
                                str(results_dir / "eigenvalue_spectrum.png"))

    print(f"\nAll plots saved to {results_dir}/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate PGDR figures")
    parser.add_argument("--results", type=str, default="pgdr/results",
                        help="Directory with result JSON/npy files")
    args = parser.parse_args()

    generate_all_plots(args.results)

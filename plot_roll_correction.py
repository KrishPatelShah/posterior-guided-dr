"""
Covariance calibration plot showing roll joint miscalibration and the
effect of manually inflating their Sigma diagonal entries by 8×.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("pgdr/results/20260408_102450_friction")

ROLL_PARAMS = {
    "friction_Left_Hip_Roll",
    "friction_Right_Hip_Roll",
    "friction_Left_Ankle_Roll",
}
INFLATE = 8.0

cal = json.loads((RESULTS_DIR / "calibration.json").read_text())
per_param = cal["per_param"]
pearson   = cal["pearson_correlation"]
spearman  = cal["spearman_correlation"]

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

fig, ax = plt.subplots(figsize=(7, 5.5))

normal  = [e for e in per_param if e["name"] not in ROLL_PARAMS]
rolls   = [e for e in per_param if e["name"] in ROLL_PARAMS]

# Normal points
ax.scatter(
    [e["uncertainty"] for e in normal],
    [e["sq_error"]    for e in normal],
    c="#E24A33", s=30, alpha=0.7, zorder=3,
    edgecolors="white", linewidths=0.3,
    label="Friction (other)",
)

# Roll params — original position
ax.scatter(
    [e["uncertainty"] for e in rolls],
    [e["sq_error"]    for e in rolls],
    c="#E24A33", s=60, alpha=0.9, zorder=4,
    edgecolors="black", linewidths=1.0,
    label="Roll joints (original Σ)",
)

# Roll params — corrected position (x * INFLATE, y unchanged)
ax.scatter(
    [e["uncertainty"] * INFLATE for e in rolls],
    [e["sq_error"]              for e in rolls],
    c="#2ca02c", s=60, alpha=0.9, zorder=4,
    edgecolors="black", linewidths=1.0,
    marker="D",
    label=f"Roll joints (Σ × {INFLATE:.0f})",
)

# Arrows connecting original → corrected
for e in rolls:
    x0 = e["uncertainty"]
    x1 = e["uncertainty"] * INFLATE
    y  = e["sq_error"]
    ax.annotate(
        "",
        xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(
            arrowstyle="->",
            color="black",
            lw=1.2,
            connectionstyle="arc3,rad=0.0",
        ),
        zorder=5,
    )

# Short param labels next to roll points
SHORT = {
    "friction_Left_Hip_Roll":   "L.Hip Roll",
    "friction_Right_Hip_Roll":  "R.Hip Roll",
    "friction_Left_Ankle_Roll": "L.Ankle Roll",
}
for e in rolls:
    ax.annotate(
        SHORT[e["name"]],
        xy=(e["uncertainty"], e["sq_error"]),
        xytext=(-4, 6), textcoords="offset points",
        fontsize=7, color="black", ha="right",
    )

# Trend line from original data (log-log)
x_all = np.array([e["uncertainty"] for e in per_param])
y_all = np.array([e["sq_error"]    for e in per_param])
mask  = (x_all > 0) & (y_all > 0)
z     = np.polyfit(np.log(x_all[mask]), np.log(y_all[mask]), 1)
x_line = np.linspace(x_all[mask].min(), x_all[mask].max() * INFLATE * 1.2, 200)
ax.plot(x_line, np.exp(z[1]) * x_line ** z[0],
        "--", color="gray", alpha=0.5, linewidth=1, label="Trend (original fit)")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\mathrm{diag}(\Sigma)_i$ (CMA-ES uncertainty)")
ax.set_ylabel(r"$(p^*_i - p^{\mathrm{true}}_i)^2$ (identification error)")
ax.set_title(
    f"Covariance Calibration — Roll Joint Correction\n"
    f"Pearson r = {pearson:.3f}, Spearman ρ = {spearman:.3f}  "
    f"(original)   ×{INFLATE:.0f} inflation → arrows"
)
ax.legend(loc="upper left", framealpha=0.9)

out = RESULTS_DIR / "figures" / "covariance_calibration_roll_correction.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out))
plt.close(fig)
print(f"Saved to {out}")

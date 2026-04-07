"""
One-at-a-time sensitivity analysis for parameter space reduction.

Before full CMA-ES identification, we perturb each parameter by ±20%
and measure trajectory divergence from the unperturbed baseline.
Parameters with negligible effect on trajectories are dropped, reducing
dimension d and improving CMA-ES convergence.

All perturbations are batched as separate MJX environments for GPU
parallelism.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mujoco
import mujoco_warp
import warp as wp
import numpy as np

from pgdr.param_space import ParamSpace, build_t1_param_space
from pgdr.sysid import _warp_model_with_params, _rollout_warp


def _traj_to_dict(q: np.ndarray, qdot: np.ndarray) -> dict:
    return {"q": q, "qdot": qdot}


def compute_trajectory_divergence(
    traj_perturbed: dict,
    traj_baseline: dict,
    w_q: float = 1.0,
    w_qdot: float = 0.1,
) -> float:
    q_err    = np.mean((traj_perturbed["q"]    - traj_baseline["q"])    ** 2)
    qdot_err = np.mean((traj_perturbed["qdot"] - traj_baseline["qdot"]) ** 2)
    return float(w_q * q_err + w_qdot * qdot_err)


def run_sensitivity_analysis(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    actions: np.ndarray,
    perturbation: float = 0.2,
    n_substeps: int = 10,
    rng_seed: int = 0,
    w_q: float = 1.0,
    w_qdot: float = 0.1,
) -> dict:
    """
    One-at-a-time perturbation study using mujoco_warp parallel worlds.

    Builds 2*d+1 perturbation vectors (baseline + ± per param), runs them
    all in one batched warp rollout, then scores by trajectory divergence.
    """
    wp.init()
    d = param_space.d
    actions_np = np.array(actions)

    # Build perturbation matrix: row 0 = baseline (zeros),
    # rows 1..d = +perturbation on param i, rows d+1..2d = -perturbation
    perturb_vecs = np.zeros((2 * d + 1, d))
    for i in range(d):
        perturb_vecs[1 + i,     i] =  perturbation
        perturb_vecs[1 + d + i, i] = -perturbation

    nworld = len(perturb_vecs)

    # Single batched warp rollout — all (2d+1) worlds in parallel
    warp_model = mujoco_warp.put_model(mj_model)
    warp_model = _warp_model_with_params(warp_model, param_space, perturb_vecs)
    warp_data  = mujoco_warp.make_data(mj_model, nworld=nworld)

    q_traj, qdot_traj = _rollout_warp(warp_model, warp_data, actions_np, n_substeps)
    # q_traj: [nworld, T, nq]

    baseline_traj = _traj_to_dict(q_traj[0], qdot_traj[0])

    pos_div = np.array([
        compute_trajectory_divergence(
            _traj_to_dict(q_traj[1 + i], qdot_traj[1 + i]),
            baseline_traj, w_q=w_q, w_qdot=w_qdot,
        )
        for i in range(d)
    ])
    neg_div = np.array([
        compute_trajectory_divergence(
            _traj_to_dict(q_traj[1 + d + i], qdot_traj[1 + d + i]),
            baseline_traj, w_q=w_q, w_qdot=w_qdot,
        )
        for i in range(d)
    ])

    sensitivity_scores = np.maximum(pos_div, neg_div)
    ranking = np.argsort(-sensitivity_scores)

    return {
        "sensitivity_scores": {param_space.params[i].name: float(sensitivity_scores[i]) for i in range(d)},
        "ranking": [
            {
                "rank": int(r + 1),
                "param_index": int(ranking[r]),
                "param_name": param_space.params[int(ranking[r])].name,
                "group": param_space.params[int(ranking[r])].group,
                "score": float(sensitivity_scores[int(ranking[r])]),
            }
            for r in range(d)
        ],
        "perturbation": perturbation,
        "total_params": d,
        "positive_divergences": {param_space.params[i].name: float(pos_div[i]) for i in range(d)},
        "negative_divergences": {param_space.params[i].name: float(neg_div[i]) for i in range(d)},
    }


def reduce_param_space(
    param_space: ParamSpace,
    sensitivity_results: dict,
    threshold: Optional[float] = None,
    keep_top_k: Optional[int] = None,
) -> tuple[ParamSpace, list[int]]:
    """
    Reduce the parameter space by dropping insensitive parameters.

    Args:
        param_space:          Full parameter space.
        sensitivity_results:  Output from run_sensitivity_analysis.
        threshold:            Drop params with score below this.
                              If None, use adaptive threshold (mean - 1 std).
        keep_top_k:           Alternative: keep exactly the top K params.

    Returns:
        (reduced_ParamSpace, kept_indices)
    """
    scores = np.array([
        sensitivity_results["sensitivity_scores"][p.name]
        for p in param_space.params
    ])

    if keep_top_k is not None:
        kept = sorted(np.argsort(-scores)[:keep_top_k].tolist())
    else:
        if threshold is None:
            # Adaptive: keep params above (mean - 1 std), but at least 50%
            mean_s = float(np.mean(scores))
            std_s = float(np.std(scores))
            threshold = max(mean_s - std_s, float(np.sort(scores)[len(scores) // 2]))
        kept = [i for i in range(len(scores)) if float(scores[i]) >= threshold]

    return param_space.select(kept), kept


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sensitivity analysis for T1 params")
    parser.add_argument("--model-xml", type=str, required=True,
                        help="Path to T1 MuJoCo XML")
    parser.add_argument("--actions", type=str, default=None,
                        help="Path to .npy action sequence [T, nu]. "
                             "If omitted, uses random actions.")
    parser.add_argument("--perturbation", type=float, default=0.2)
    parser.add_argument("--output", type=str,
                        default="pgdr/results/sensitivity_ranking.json")
    parser.add_argument("--keep-top-k", type=int, default=None,
                        help="Keep only top K sensitive params")
    args = parser.parse_args()

    mj_model = mujoco.MjModel.from_xml_path(args.model_xml)
    ps = build_t1_param_space(mj_model)

    # Load or generate actions
    if args.actions:
        actions = np.load(args.actions)
    else:
        # Generate a diverse action sequence: 10 seconds at 50 Hz
        T = 500
        # Sinusoidal actions across joints to excite all modes
        t = np.linspace(0, 10 * np.pi, T)
        freqs = np.linspace(0.5, 3.0, mj_model.nu)
        actions = 0.3 * np.sin(t[:, None] * freqs[None, :])

    print(f"Running sensitivity analysis: {ps.d} parameters, "
          f"perturbation={args.perturbation}")
    results = run_sensitivity_analysis(
        mj_model, ps, actions,
        perturbation=args.perturbation,
    )

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Saved sensitivity ranking to {out_path}")

    # Print top 20
    print("\nTop 20 most sensitive parameters:")
    for entry in results["ranking"][:20]:
        print(f"  {entry['rank']:3d}. {entry['param_name']:40s} "
              f"({entry['group']:10s})  score={entry['score']:.6f}")

    # Reduce
    if args.keep_top_k:
        reduced_ps, kept = reduce_param_space(ps, results, keep_top_k=args.keep_top_k)
        print(f"\nReduced: {ps.d} → {reduced_ps.d} parameters")
        reduced_ps.save(out_path.parent / "reduced_param_space.json")

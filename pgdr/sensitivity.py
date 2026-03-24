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

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from pgdr.param_space import ParamSpace, build_t1_param_space


@jax.jit
def _rollout_trajectory(
    model: mjx.Model,
    data: mjx.Data,
    actions: jnp.ndarray,
    n_steps: int,
) -> dict[str, jnp.ndarray]:
    """
    Open-loop rollout: apply a fixed action sequence and record trajectory.

    Args:
        model:   MJX model (possibly with perturbed parameters).
        data:    MJX data (initial state).
        actions: [T, nu] action sequence.
        n_steps: Number of physics substeps per control step.

    Returns:
        Dictionary with 'q' [T, nq] and 'qdot' [T, nv].
    """
    def step_fn(carry, action):
        data = carry
        # Apply action to data.ctrl
        data = data.replace(ctrl=action)
        # Step physics n_steps times
        def substep(d, _):
            return mjx.step(model, d), None
        data, _ = jax.lax.scan(substep, data, None, length=n_steps)
        return data, jnp.concatenate([data.qpos, data.qvel])

    _, trajectory = jax.lax.scan(step_fn, data, actions)

    nq = model.nq
    return {
        "q": trajectory[:, :nq],
        "qdot": trajectory[:, nq:],
    }


def compute_trajectory_divergence(
    traj_perturbed: dict[str, jnp.ndarray],
    traj_baseline: dict[str, jnp.ndarray],
    w_q: float = 1.0,
    w_qdot: float = 0.1,
) -> float:
    """
    MSE between a perturbed trajectory and the unperturbed baseline.

    This is the same loss function used in sys-id (Eq. 1 of the proposal),
    ensuring sensitivity is measured in the same metric space.
    """
    q_err = jnp.mean((traj_perturbed["q"] - traj_baseline["q"]) ** 2)
    qdot_err = jnp.mean((traj_perturbed["qdot"] - traj_baseline["qdot"]) ** 2)
    return w_q * q_err + w_qdot * qdot_err


def run_sensitivity_analysis(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    actions: jnp.ndarray,
    perturbation: float = 0.2,
    n_substeps: int = 10,
    rng_seed: int = 0,
) -> dict:
    """
    One-at-a-time perturbation study.

    For each parameter i:
        - Set p[i] = +perturbation (in normalized space), all others at 0
        - Roll out, compute divergence from baseline
        - Repeat with p[i] = -perturbation
        - Record max divergence across ± directions

    All 2*d perturbations are batched into a single vectorized rollout.

    Args:
        mj_model:     MuJoCo model (CPU).
        param_space:  Full (unreduced) parameter space.
        actions:      [T, nu] reference action sequence.
        perturbation: Perturbation magnitude in normalized units (0.2 = 20%).
        n_substeps:   Physics substeps per control step.
        rng_seed:     For initial state.

    Returns:
        Dictionary with sensitivity scores and ranking.
    """
    d = param_space.d
    rng = jax.random.PRNGKey(rng_seed)

    # Put MJX model on device
    mjx_model_default = mjx.put_model(mj_model)
    mjx_data_default = mjx.put_data(mj_model, mujoco.MjData(mj_model))

    # --- Baseline rollout (all params at default) ---
    baseline_traj = _rollout_trajectory(
        mjx_model_default, mjx_data_default, actions, n_substeps
    )

    # --- Build perturbation vectors: [2*d, d] ---
    # First d rows: +perturbation on each param
    # Next d rows: -perturbation on each param
    perturbation_vecs = jnp.zeros((2 * d, d))
    perturbation_vecs = perturbation_vecs.at[jnp.arange(d), jnp.arange(d)].set(perturbation)
    perturbation_vecs = perturbation_vecs.at[d + jnp.arange(d), jnp.arange(d)].set(-perturbation)

    # --- Batch rollout of all perturbations ---
    def rollout_one_perturbation(p_norm):
        """Rollout with a single perturbed parameter vector."""
        model_perturbed = param_space.inject(mjx_model_default, p_norm)
        traj = _rollout_trajectory(model_perturbed, mjx_data_default, actions, n_substeps)
        return compute_trajectory_divergence(traj, baseline_traj)

    # vmap over all perturbation vectors
    all_divergences = jax.vmap(rollout_one_perturbation)(perturbation_vecs)

    # --- Compute sensitivity scores ---
    # Take the max of +/- perturbation for each parameter
    pos_divergences = all_divergences[:d]
    neg_divergences = all_divergences[d:]
    sensitivity_scores = jnp.maximum(pos_divergences, neg_divergences)

    # --- Rank parameters ---
    ranking = jnp.argsort(-sensitivity_scores)  # Descending

    results = {
        "sensitivity_scores": {
            param_space.params[i].name: float(sensitivity_scores[i])
            for i in range(d)
        },
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
        "positive_divergences": {
            param_space.params[i].name: float(pos_divergences[i])
            for i in range(d)
        },
        "negative_divergences": {
            param_space.params[i].name: float(neg_divergences[i])
            for i in range(d)
        },
    }

    return results


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
    scores = jnp.array([
        sensitivity_results["sensitivity_scores"][p.name]
        for p in param_space.params
    ])

    if keep_top_k is not None:
        kept = jnp.argsort(-scores)[:keep_top_k]
        kept = sorted(kept.tolist())
    else:
        if threshold is None:
            # Adaptive: keep params above (mean - 1 std), but at least 50%
            mean_s = float(jnp.mean(scores))
            std_s = float(jnp.std(scores))
            threshold = max(mean_s - std_s, float(jnp.sort(scores)[len(scores) // 2]))
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
        actions = jnp.load(args.actions)
    else:
        # Generate a diverse action sequence: 10 seconds at 50 Hz
        T = 500
        rng = jax.random.PRNGKey(0)
        # Sinusoidal actions across joints to excite all modes
        t = jnp.linspace(0, 10 * jnp.pi, T)
        freqs = jnp.linspace(0.5, 3.0, mj_model.nu)
        actions = 0.3 * jnp.sin(t[:, None] * freqs[None, :])

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

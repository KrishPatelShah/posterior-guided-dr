"""
Evaluation pipeline for PGDR experiments.

Computes all metrics from the proposal (Section 4):
    Primary:   RMS velocity tracking error (nominal + perturbed)
    Secondary: Parameter recovery error, covariance calibration,
               trajectory prediction error, robustness under perturbation

The critical diagnostic is the covariance calibration plot:
    x = diag(Σ)[i]         (how uncertain CMA-ES was about param i)
    y = (p*[i] - p_true[i])²  (how wrong the identification was about param i)
    A positive correlation validates PGDR's core assumption.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import warnings

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from pgdr.param_space import ParamSpace, build_t1_param_space, inject_contact_params_to_all_feet
from pgdr.sysid import ReferenceTrajectory


def _sanitize_geom_types_for_mjx(mj_model: mujoco.MjModel) -> int:
    """Convert cylinder geoms to capsules for broader MJX collision support."""
    cyl_type = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    cap_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    converted = 0
    for gid in range(mj_model.ngeom):
        if int(mj_model.geom_type[gid]) == cyl_type:
            mj_model.geom_type[gid] = cap_type
            converted += 1
    return converted


def _put_model_with_mjx_fallback(mj_model: mujoco.MjModel) -> mjx.Model:
    """Create MJX model, retrying once after geometry sanitization if needed."""
    try:
        return mjx.put_model(mj_model)
    except NotImplementedError as e:
        if "collisions not implemented" not in str(e):
            raise
        converted = _sanitize_geom_types_for_mjx(mj_model)
        if converted <= 0:
            raise RuntimeError(
                "MJX rejected model collisions, and no cylinder geoms were found "
                "to auto-convert."
            ) from e
        warnings.warn(
            f"MJX collision workaround applied: converted {converted} cylinder "
            "geoms to capsules.",
            RuntimeWarning,
            stacklevel=2,
        )
        return mjx.put_model(mj_model)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EvalResults:
    """Aggregated evaluation results for one condition."""
    condition: str
    seed: int

    # Primary metric: RMS velocity tracking error
    rms_vel_error_nominal: float = 0.0
    rms_vel_error_perturbed_payload: float = 0.0
    rms_vel_error_perturbed_friction: float = 0.0

    # Secondary (sim-to-sim only)
    param_recovery_error: float = 0.0
    covariance_calibration_corr: float = 0.0
    held_out_traj_error: float = 0.0

    # Per-group breakdown
    per_group_recovery: Optional[dict] = None
    per_param_calibration: Optional[dict] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ---------------------------------------------------------------------------
# Velocity tracking evaluation
# ---------------------------------------------------------------------------

def evaluate_velocity_tracking(
    policy_fn,
    mjx_model: mjx.Model,
    command_sequence: list[dict],
    control_dt: float,
    n_substeps: int = 10,
    num_episodes: int = 50,
    rng_seed: int = 0,
) -> dict:
    """
    Evaluate a policy's velocity tracking performance.

    Args:
        policy_fn:         Callable(obs, rng) -> action.
        mjx_model:         MJX model (with appropriate physical parameters).
        command_sequence:   List of {vx, vy, wz, duration} commands.
        control_dt:        Control timestep.
        n_substeps:        Physics substeps per control step.
        num_episodes:      Number of evaluation episodes.
        rng_seed:          Random seed.

    Returns:
        Dictionary with per-axis and total RMS velocity errors.
    """
    rng = jax.random.PRNGKey(rng_seed)

    @jax.jit
    def run_episode(rng):
        rng, init_rng = jax.random.split(rng)
        mjx_data = mjx.make_data(mjx_model)

        errors_vx = []
        errors_vy = []
        errors_wz = []

        for cmd in command_sequence:
            n_steps = int(cmd["duration"] / control_dt)
            target_vx = cmd.get("vx", 0.0)
            target_vy = cmd.get("vy", 0.0)
            target_wz = cmd.get("wz", 0.0)

            def step_fn(carry, _):
                data, rng = carry
                rng, action_rng = jax.random.split(rng)

                cmd_vec = jnp.array([target_vx, target_vy, target_wz])
                qpos_obs = jnp.concatenate([data.qpos[2:7], data.qpos[7:]])
                obs = jnp.concatenate([qpos_obs, data.qvel, cmd_vec])
                action = policy_fn(obs, action_rng)
                data = data.replace(ctrl=action * 0.25)

                def substep(d, _):
                    return mjx.step(mjx_model, d), None
                data, _ = jax.lax.scan(substep, data, None, length=n_substeps)

                # Base velocity in world frame (first 3 entries of qvel for floating base)
                actual_vx = data.qvel[0]
                actual_vy = data.qvel[1]
                actual_wz = data.qvel[5]  # Yaw rate

                err = jnp.array([
                    (actual_vx - target_vx) ** 2,
                    (actual_vy - target_vy) ** 2,
                    (actual_wz - target_wz) ** 2,
                ])
                return (data, rng), err

            (mjx_data, rng), errs = jax.lax.scan(
                step_fn, (mjx_data, rng), None, length=n_steps
            )
            errors_vx.append(errs[:, 0])
            errors_vy.append(errs[:, 1])
            errors_wz.append(errs[:, 2])

        all_vx = jnp.concatenate(errors_vx)
        all_vy = jnp.concatenate(errors_vy)
        all_wz = jnp.concatenate(errors_wz)

        return {
            "rms_vx": jnp.sqrt(jnp.mean(all_vx)),
            "rms_vy": jnp.sqrt(jnp.mean(all_vy)),
            "rms_wz": jnp.sqrt(jnp.mean(all_wz)),
            "rms_total": jnp.sqrt(jnp.mean(all_vx + all_vy + all_wz)),
        }

    # Run multiple episodes
    rngs = jax.random.split(rng, num_episodes)
    results = jax.vmap(run_episode)(rngs)

    return {
        "rms_vx": float(jnp.mean(results["rms_vx"])),
        "rms_vy": float(jnp.mean(results["rms_vy"])),
        "rms_wz": float(jnp.mean(results["rms_wz"])),
        "rms_total": float(jnp.mean(results["rms_total"])),
        "rms_total_std": float(jnp.std(results["rms_total"])),
    }


# ---------------------------------------------------------------------------
# Parameter recovery (sim-to-sim)
# ---------------------------------------------------------------------------

def compute_param_recovery(
    p_star: jnp.ndarray,
    p_true: jnp.ndarray,
    param_space: ParamSpace,
) -> dict:
    """
    Compute per-parameter and per-group recovery error.

    Args:
        p_star:  [d] identified parameters (normalized).
        p_true:  [d] ground truth parameters (normalized).
        param_space: Parameter space definition.

    Returns:
        Dictionary with total error and per-group breakdown.
    """
    per_param_error = (p_star - p_true) ** 2
    total_error = float(jnp.sqrt(jnp.sum(per_param_error)))

    # Per-group
    groups = {}
    for group_name in ["friction", "mass", "actuator", "contact"]:
        indices = param_space.group_indices(group_name)
        if indices:
            group_err = float(jnp.sqrt(jnp.sum(per_param_error[jnp.array(indices)])))
            group_mean = float(jnp.mean(jnp.sqrt(per_param_error[jnp.array(indices)])))
            groups[group_name] = {
                "rmse": group_err,
                "mean_per_param": group_mean,
                "num_params": len(indices),
            }

    return {
        "total_rmse": total_error,
        "per_group": groups,
        "per_param_sq_error": per_param_error.tolist(),
    }


# ---------------------------------------------------------------------------
# Covariance calibration (core PGDR diagnostic)
# ---------------------------------------------------------------------------

def compute_covariance_calibration(
    p_star: jnp.ndarray,
    p_true: jnp.ndarray,
    Sigma: jnp.ndarray,
    param_space: ParamSpace,
) -> dict:
    """
    The key diagnostic validating PGDR's core assumption.

    Computes the correlation between:
        - diag(Σ)[i]:  how uncertain CMA-ES was about parameter i
        - (p*[i] - p_true[i])²:  how wrong the identification was

    A positive correlation means the covariance correctly tracks where
    the identification is unreliable — the foundation of PGDR.

    Args:
        p_star:       [d] identified parameters (normalized).
        p_true:       [d] ground truth (normalized).
        Sigma:        [d, d] covariance matrix.
        param_space:  Parameter space definition.

    Returns:
        Dictionary with correlation, p-value, per-param data.
    """
    d = len(p_star)

    uncertainty = jnp.diag(Sigma)                  # Per-param variance
    sq_error = (p_star - p_true) ** 2               # Per-param squared error

    # Pearson correlation
    corr = _pearson_correlation(uncertainty, sq_error)

    # Spearman rank correlation (more robust)
    rank_unc = _rank(uncertainty)
    rank_err = _rank(sq_error)
    spearman = _pearson_correlation(rank_unc, rank_err)

    # Per-group correlations
    group_calibration = {}
    for group_name in ["friction", "mass", "actuator", "contact"]:
        indices = param_space.group_indices(group_name)
        if len(indices) >= 3:  # Need at least 3 points for correlation
            idx = jnp.array(indices)
            g_corr = _pearson_correlation(uncertainty[idx], sq_error[idx])
            group_calibration[group_name] = float(g_corr)

    # Data for plotting
    per_param = []
    for i in range(d):
        per_param.append({
            "name": param_space.params[i].name,
            "group": param_space.params[i].group,
            "uncertainty": float(uncertainty[i]),
            "sq_error": float(sq_error[i]),
        })

    return {
        "pearson_correlation": float(corr),
        "spearman_correlation": float(spearman),
        "per_group_correlation": group_calibration,
        "per_param": per_param,
        "interpretation": _interpret_calibration(float(corr)),
    }


def _pearson_correlation(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Compute Pearson correlation coefficient."""
    x_centered = x - jnp.mean(x)
    y_centered = y - jnp.mean(y)
    num = jnp.sum(x_centered * y_centered)
    den = jnp.sqrt(jnp.sum(x_centered ** 2) * jnp.sum(y_centered ** 2))
    return jnp.where(den > 1e-10, num / den, 0.0)


def _rank(x: jnp.ndarray) -> jnp.ndarray:
    """Compute ranks (1-based) for Spearman correlation."""
    order = jnp.argsort(x)
    ranks = jnp.zeros_like(x)
    ranks = ranks.at[order].set(jnp.arange(len(x), dtype=x.dtype) + 1)
    return ranks


def _interpret_calibration(corr: float) -> str:
    """Interpret the calibration correlation for the paper."""
    if corr > 0.5:
        return ("Strong positive correlation: CMA-ES covariance reliably "
                "tracks identification uncertainty. PGDR is well-motivated.")
    elif corr > 0.2:
        return ("Moderate positive correlation: CMA-ES covariance partially "
                "reflects uncertainty. PGDR should provide benefit over isotropic DR.")
    elif corr > -0.1:
        return ("Weak/no correlation: CMA-ES covariance does not strongly "
                "track identification error. PGDR may not outperform isotropic DR.")
    else:
        return ("Negative correlation: unexpected. The covariance may be "
                "miscalibrated. Consider using empirical covariance instead.")


# ---------------------------------------------------------------------------
# Trajectory prediction error
# ---------------------------------------------------------------------------

def compute_trajectory_prediction_error(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_star: jnp.ndarray,
    held_out_ref: ReferenceTrajectory,
    n_substeps: int = 10,
    w_q: float = 1.0,
    w_qdot: float = 0.1,
) -> float:
    """
    Evaluate how well p* predicts held-out reference trajectories
    (not used during identification).
    """
    if held_out_ref.actions.ndim != 2 or held_out_ref.actions.shape[1] != mj_model.nu:
        warnings.warn(
            f"Held-out actions shape {tuple(held_out_ref.actions.shape)} does not "
            f"match expected [T, {mj_model.nu}].",
            RuntimeWarning,
            stacklevel=2,
        )
    mjx_model = _put_model_with_mjx_fallback(mj_model)
    mjx_model = param_space.inject(mjx_model, p_star)
    mjx_data = mjx.put_data(mj_model, mujoco.MjData(mj_model))

    def step_fn(data, action):
        data = data.replace(ctrl=action)
        def substep(d, _):
            return mjx.step(mjx_model, d), None
        data, _ = jax.lax.scan(substep, data, None, length=n_substeps)
        return data, jnp.concatenate([data.qpos, data.qvel])

    _, trajectory = jax.lax.scan(step_fn, mjx_data, held_out_ref.actions)

    nq = mj_model.nq
    q_sim = trajectory[:, :nq]
    qdot_sim = trajectory[:, nq:]

    error = (
        w_q * jnp.mean((q_sim - held_out_ref.q) ** 2) +
        w_qdot * jnp.mean((qdot_sim - held_out_ref.qdot) ** 2)
    )
    return float(error)


# ---------------------------------------------------------------------------
# Perturbation robustness
# ---------------------------------------------------------------------------

def apply_perturbation(
    mjx_model: mjx.Model,
    mj_model: mujoco.MjModel,
    perturbation_type: str,
) -> mjx.Model:
    """
    Apply a novel perturbation to test out-of-distribution robustness.

    Perturbation types:
        "payload_1kg":      Add 1 kg to the torso body
        "floor_friction":   Change floor friction by ±30%
        "both":             Both perturbations simultaneously
    """
    if perturbation_type in ("payload_1kg", "both"):
        # Add 1 kg to the torso (body index 1, typically)
        torso_id = 1  # Adjust based on actual T1 model
        for i in range(mj_model.nbody):
            name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and "torso" in name.lower():
                torso_id = i
                break

        new_mass = mjx_model.body_mass.at[torso_id].add(1.0)
        mjx_model = mjx_model.replace(body_mass=new_mass)

    if perturbation_type in ("floor_friction", "both"):
        # Reduce floor friction by 30%
        # Floor geom is typically geom 0
        new_friction = mjx_model.geom_friction.at[0, 0].multiply(0.7)
        mjx_model = mjx_model.replace(geom_friction=new_friction)

    return mjx_model


# ---------------------------------------------------------------------------
# Full evaluation suite
# ---------------------------------------------------------------------------

def evaluate_all_conditions(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_star: jnp.ndarray,
    Sigma: jnp.ndarray,
    p_true: Optional[jnp.ndarray],
    checkpoints_dir: str,
    command_sequence: list[dict],
    held_out_ref: Optional[ReferenceTrajectory] = None,
    num_episodes: int = 50,
    perturbation_type: Optional[str] = None,
) -> dict:
    """
    Run the full evaluation suite across all conditions.

    Args:
        mj_model:          Base MuJoCo model.
        param_space:       Parameter space.
        p_star:            Identified parameters.
        Sigma:             CMA-ES covariance.
        p_true:            Ground truth (sim-to-sim only, None for real).
        checkpoints_dir:   Directory with trained policies per condition.
        command_sequence:   Evaluation command sequence.
        held_out_ref:      Held-out reference trajectory (optional).
        num_episodes:      Episodes per evaluation.

    Returns:
        Dictionary of all results.
    """
    results = {}

    # --- Identification diagnostics (sim-to-sim only) ---
    if p_true is not None:
        results["param_recovery"] = compute_param_recovery(
            p_star, p_true, param_space
        )
        results["covariance_calibration"] = compute_covariance_calibration(
            p_star, p_true, Sigma, param_space
        )
        print(f"Parameter recovery RMSE: "
              f"{results['param_recovery']['total_rmse']:.4f}")
        print(f"Covariance calibration (Pearson): "
              f"{results['covariance_calibration']['pearson_correlation']:.3f}")
        print(f"  {results['covariance_calibration']['interpretation']}")

    # --- Held-out trajectory prediction ---
    if held_out_ref is not None:
        results["held_out_traj_error"] = compute_trajectory_prediction_error(
            mj_model, param_space, p_star, held_out_ref
        )
        print(f"Held-out trajectory error: {results['held_out_traj_error']:.6f}")

    # --- Covariance spectrum analysis ---
    eigvals = jnp.linalg.eigvalsh(Sigma)
    results["covariance_spectrum"] = {
        "eigenvalues": sorted([float(e) for e in eigvals], reverse=True),
        "condition_number": float(jnp.max(eigvals) / jnp.max(jnp.array([jnp.min(eigvals), 1e-10]))),
        "effective_rank": int(jnp.sum(eigvals > 0.01 * jnp.max(eigvals))),
        "trace": float(jnp.trace(Sigma)),
    }

    # --- Per-condition policy evaluation ---
    from pgdr._ppo import PPOAgent, PPOConfig

    mjx_model_default = _put_model_with_mjx_fallback(mj_model)

    # Inject p* into model for evaluation
    mjx_model_eval = param_space.inject(mjx_model_default, p_star)

    checkpoints = Path(checkpoints_dir)
    if checkpoints.exists():
        for condition_dir in sorted(checkpoints.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition_name = condition_dir.name

            # Try best.pkl first, then final.pkl
            policy_path = condition_dir / "best.pkl"
            if not policy_path.exists():
                policy_path = condition_dir / "final.pkl"
            if not policy_path.exists():
                print(f"  Skipping {condition_name} (no checkpoint)")
                continue

            print(f"\nEvaluating: {condition_name}")

            # Load agent
            obs_dim = (mj_model.nq - 2) + mj_model.nv + 3
            act_dim = mj_model.nu
            agent = PPOAgent(PPOConfig(), obs_dim, act_dim, jax.random.PRNGKey(0))
            agent.load(policy_path)
            policy_fn = agent.make_policy_fn()

            # Evaluate on nominal (Sim A) parameters
            eval_model = mjx_model_eval
            if perturbation_type:
                eval_model = apply_perturbation(eval_model, mj_model, perturbation_type)

            vel_results = evaluate_velocity_tracking(
                policy_fn, eval_model, command_sequence,
                control_dt=0.02, num_episodes=num_episodes,
            )
            results[condition_name] = vel_results
            print(f"  RMS total: {vel_results['rms_total']:.4f} "
                  f"(vx={vel_results['rms_vx']:.4f}, "
                  f"vy={vel_results['rms_vy']:.4f}, "
                  f"wz={vel_results['rms_wz']:.4f})")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PGDR Evaluation")
    subparsers = parser.add_subparsers(dest="command")

    # --- covariance-calibration ---
    p_cal = subparsers.add_parser("covariance-calibration",
                                   help="Check covariance calibration (sim-to-sim)")
    p_cal.add_argument("--p-star", type=str, required=True)
    p_cal.add_argument("--sigma", type=str, required=True)
    p_cal.add_argument("--p-true", type=str, required=True)
    p_cal.add_argument("--param-space", type=str, default=None)
    p_cal.add_argument("--model-xml", type=str, required=True)
    p_cal.add_argument("--output", type=str,
                       default="pgdr/results/calibration.json")

    # --- sim2sim ---
    p_sim = subparsers.add_parser("sim2sim",
                                   help="Sim-to-sim evaluation of all conditions")
    p_sim.add_argument("--model-xml", type=str, required=True)
    p_sim.add_argument("--p-star", type=str, required=True)
    p_sim.add_argument("--sigma", type=str, required=True)
    p_sim.add_argument("--p-true", type=str, required=True)
    p_sim.add_argument("--checkpoints", type=str, default="pgdr/checkpoints")
    p_sim.add_argument("--param-space", type=str, default=None)
    p_sim.add_argument("--perturbation", type=str, default=None,
                       choices=["payload_1kg", "floor_friction", "both"])
    p_sim.add_argument("--output", type=str,
                       default="pgdr/results/eval_results.json")

    args = parser.parse_args()

    if args.command == "covariance-calibration":
        mj_model = mujoco.MjModel.from_xml_path(args.model_xml)
        if args.param_space:
            ps = ParamSpace.load(args.param_space)
        else:
            ps = build_t1_param_space(mj_model)

        p_star = jnp.load(args.p_star)
        Sigma = jnp.load(args.sigma)
        p_true = jnp.load(args.p_true)

        cal = compute_covariance_calibration(p_star, p_true, Sigma, ps)

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cal, indent=2))

        print(f"Pearson correlation:  {cal['pearson_correlation']:.3f}")
        print(f"Spearman correlation: {cal['spearman_correlation']:.3f}")
        print(f"Interpretation: {cal['interpretation']}")
        print(f"\nPer-group correlations:")
        for group, corr in cal["per_group_correlation"].items():
            print(f"  {group:12s}: {corr:.3f}")
        print(f"\nSaved to {out}")

    elif args.command == "sim2sim":
        mj_model = mujoco.MjModel.from_xml_path(args.model_xml)
        if args.param_space:
            ps = ParamSpace.load(args.param_space)
        else:
            ps = build_t1_param_space(mj_model)

        p_star = jnp.load(args.p_star)
        Sigma = jnp.load(args.sigma)
        p_true = jnp.load(args.p_true)

        commands = [
            {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
            {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
            {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
            {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
        ]

        results = evaluate_all_conditions(
            mj_model, ps, p_star, Sigma, p_true,
            args.checkpoints, commands,
            perturbation_type=args.perturbation,
        )

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved results to {out}")

    else:
        parser.print_help()

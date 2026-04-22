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

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np

from pgdr.model_utils import load_mj_model
from pgdr.param_space import ParamSpace, build_t1_param_space, inject_contact_params_to_all_feet
from pgdr.sysid import ReferenceTrajectory
from pgdr.t1_env import _build_obs, _get_default_qpos


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

def _make_episode_fn(policy_fn, command_sequence, control_dt, default_joint_pos, n_substeps):
    """
    Returns a JIT-compiled run_episode(model, rng) -> metrics dict.

    Factored out so the compiled function can be reused across many models
    without recompilation (model is a traced arg, not static).
    """
    @jax.jit
    def run_episode(model, rng):
        _, init_rng = jax.random.split(rng)
        mjx_data = mjx.make_data(model)

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
                command = jnp.array([target_vx, target_vy, target_wz], dtype=data.qpos.dtype)
                obs = _build_obs(data, command, model, default_joint_pos)
                action = policy_fn(obs, action_rng)
                data = data.replace(ctrl=action)

                def substep(d, _):
                    return mjx.step(model, d), None
                data, _ = jax.lax.scan(substep, data, None, length=n_substeps)

                actual_vx = data.qvel[0]
                actual_vy = data.qvel[1]
                actual_wz = data.qvel[5]
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

    return run_episode


def evaluate_velocity_tracking(
    policy_fn,
    mjx_model: mjx.Model,
    command_sequence: list[dict],
    control_dt: float,
    default_joint_pos: jnp.ndarray,
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
    run_episode = _make_episode_fn(
        policy_fn, command_sequence, control_dt, default_joint_pos, n_substeps
    )
    rngs = jax.random.split(rng, num_episodes)
    results = jax.vmap(functools.partial(run_episode, mjx_model))(rngs)

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
    mjx_model = mjx.put_model(mj_model)
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
    default_joint_pos = jnp.array(_get_default_qpos(mj_model)[7:])
    mj_model_mjx = mjx.put_model(mj_model)
    checkpoints = Path(checkpoints_dir)
    if checkpoints.exists():
        for condition_dir in sorted(checkpoints.iterdir()):
            if not condition_dir.is_dir():
                continue
            condition_name = condition_dir.name
            print(f"\nEvaluating: {condition_name}")

            # Load policy
            policy_path = condition_dir / "final.pkl"
            if not policy_path.exists():
                print(f"  Skipping (no final.pkl)")
                continue

            # Load PPO agent and evaluate
            try:
                from pgdr._ppo import PPOAgent, PPOConfig
                from pgdr.t1_env import OBS_DIM, ACT_DIM

                rng = jax.random.PRNGKey(0)
                # Load agent from checkpoint
                agent = PPOAgent.load(policy_path, rng)

                # Build eval model using p_star (nominal conditions)
                eval_mjx_model = param_space.inject(mj_model_mjx, p_star)

                # Build deterministic policy function
                def policy_fn(obs, rng_unused):
                    return agent.get_deterministic_action(obs[None]).squeeze(0)

                # Standard eval commands
                eval_commands = [
                    {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
                    {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
                    {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
                    {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
                ]

                vel_results = evaluate_velocity_tracking(
                    policy_fn=policy_fn,
                    mjx_model=eval_mjx_model,
                    command_sequence=eval_commands,
                    control_dt=0.02,
                    default_joint_pos=default_joint_pos,
                )
                results[condition_name] = vel_results
                print(f"  RMS vel error: {vel_results.get('rms_total', 'N/A'):.4f}")

            except Exception as e:
                print(f"  Evaluation failed: {e}")
                results[condition_name] = {"error": str(e)}

    return results


# ---------------------------------------------------------------------------
# Parameter perturbation sweep (core PGDR eval)
# ---------------------------------------------------------------------------

def run_param_perturbation_sweep(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_star: jnp.ndarray,
    Sigma: jnp.ndarray,
    checkpoints_dir: str,
    command_sequence: list[dict],
    alpha_levels: list[float],
    conditions_to_eval: Optional[list[str]] = None,
    n_param_samples: int = 20,
    num_episodes: int = 5,
    rng_seed: int = 42,
) -> dict:
    """
    Sweep uncertainty scale α and evaluate each policy at parameters sampled
    from N(p*, αΣ). This is the core PGDR eval: C4 should degrade gracefully
    as α increases while C2 (no DR) degrades rapidly.

    Returns:
        {
          "alpha_levels": [...],
          "<condition>": {"mean": [...], "std": [...], "seeds": [[...], ...]},
          ...
        }
    """
    from pgdr._ppo import PPOAgent
    from pgdr.t1_env import _get_default_qpos

    checkpoints = Path(checkpoints_dir)
    default_joint_pos = jnp.array(_get_default_qpos(mj_model)[7:])
    base_mjx_model = mjx.put_model(mj_model)

    # Cholesky for sampling
    L = jnp.linalg.cholesky(Sigma + 1e-8 * jnp.eye(Sigma.shape[0]))

    # Group checkpoint dirs by condition
    condition_seeds: dict[str, list[Path]] = {}
    for d in sorted(checkpoints.iterdir()):
        if not d.is_dir() or not (d / "final.pkl").exists():
            continue
        parts = d.name.rsplit("_seed", 1)
        cond = parts[0] if len(parts) == 2 and parts[1].isdigit() else d.name
        if conditions_to_eval and cond not in conditions_to_eval:
            continue
        condition_seeds.setdefault(cond, []).append(d)

    results: dict = {"alpha_levels": alpha_levels}

    for cond, seed_dirs in condition_seeds.items():
        print(f"\nSweeping: {cond} ({len(seed_dirs)} seeds)")
        seed_curves = []

        for seed_dir in seed_dirs:
            rng = jax.random.PRNGKey(rng_seed)
            agent = PPOAgent.load(seed_dir / "final.pkl", rng)

            def policy_fn(obs, _rng):
                return agent.get_deterministic_action(obs[None]).squeeze(0)

            run_episode = _make_episode_fn(
                policy_fn, command_sequence, 0.02, default_joint_pos, n_substeps=10
            )

            # Compile once per seed: vmap over (param_sample, episode).
            # inject is vmappable; run_episode traces model abstractly so
            # different concrete param values do not trigger recompilation.
            inject_batch = jax.jit(
                jax.vmap(lambda p: param_space.inject(base_mjx_model, p))
            )

            @jax.jit
            def eval_batch(batched_models, ep_rngs):
                # batched_models: (n_param_samples, ...) model pytree
                # ep_rngs:        (n_param_samples, num_episodes, 2)
                def eval_one(model, rngs_for_sample):
                    out = jax.vmap(lambda r: run_episode(model, r))(rngs_for_sample)
                    return jnp.mean(out["rms_total"])
                return jax.vmap(eval_one)(batched_models, ep_rngs)

            # Precompute clipping bounds (same for all alpha)
            norm_lower = param_space.to_normalized_vec(param_space._lowers)
            norm_upper = param_space.to_normalized_vec(param_space._uppers)
            finite_lower = jnp.where(jnp.isfinite(norm_lower), norm_lower, -1e6)
            finite_upper = jnp.where(jnp.isfinite(norm_upper), norm_upper,  1e6)

            level_errors = []
            for alpha in alpha_levels:
                rng, sample_rng, ep_rng = jax.random.split(rng, 3)
                z = jax.random.normal(sample_rng, (n_param_samples, param_space.d))
                p_samples = p_star + jnp.sqrt(alpha) * (z @ L.T)
                p_samples = jnp.clip(p_samples, finite_lower, finite_upper)

                batched_models = inject_batch(p_samples)
                ep_rngs = jax.random.split(ep_rng, n_param_samples * num_episodes)
                ep_rngs = ep_rngs.reshape(n_param_samples, num_episodes, -1)

                sample_errors = eval_batch(batched_models, ep_rngs)
                mean_err = float(jnp.mean(sample_errors))
                level_errors.append(mean_err)
                print(f"  {seed_dir.name}  α={alpha:.1f}  rms={mean_err:.4f}")
            seed_curves.append(level_errors)

        arr = np.array(seed_curves)
        results[cond] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "seeds": arr.tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# Sim-to-sim transfer test
# ---------------------------------------------------------------------------

def run_transfer_test(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_star: jnp.ndarray,
    checkpoints_dir: str,
    command_sequence: list[dict],
    delta_levels: list[float],
    conditions_to_eval: Optional[list[str]] = None,
    n_directions: int = 10,
    num_episodes: int = 5,
    rng_seed: int = 42,
) -> dict:
    """
    Fair sim-to-sim transfer test.

    Evaluates all conditions at p_eval = p_star + δ * direction, where
    direction is a random unit vector in normalized parameter space and δ
    is expressed as a fraction of each parameter's [lower, upper] range.

    Unlike the param sweep (which uses Σ for sampling), this is agnostic
    to the CMA-ES posterior and gives each condition an equal footing.

    Returns:
        {
          "delta_levels": [...],
          "<condition>": {"mean": [...], "std": [...], "seeds": [[...], ...]},
        }
    """
    from pgdr._ppo import PPOAgent
    from pgdr.t1_env import _get_default_qpos

    checkpoints = Path(checkpoints_dir)
    default_joint_pos = jnp.array(_get_default_qpos(mj_model)[7:])
    base_mjx_model = mjx.put_model(mj_model)

    # Normalized parameter range half-widths — used to scale δ
    norm_lower = param_space.to_normalized_vec(param_space._lowers)
    norm_upper = param_space.to_normalized_vec(param_space._uppers)
    finite_lower = jnp.where(jnp.isfinite(norm_lower), norm_lower, p_star - 1.0)
    finite_upper = jnp.where(jnp.isfinite(norm_upper), norm_upper, p_star + 1.0)
    param_range = finite_upper - finite_lower  # per-parameter range width

    # Group checkpoint dirs by condition
    condition_seeds: dict[str, list[Path]] = {}
    for d in sorted(checkpoints.iterdir()):
        if not d.is_dir() or not (d / "final.pkl").exists():
            continue
        parts = d.name.rsplit("_seed", 1)
        cond = parts[0] if len(parts) == 2 and parts[1].isdigit() else d.name
        if conditions_to_eval and cond not in conditions_to_eval:
            continue
        condition_seeds.setdefault(cond, []).append(d)

    # Pre-sample random unit directions (same for all conditions/seeds)
    rng = jax.random.PRNGKey(rng_seed)
    rng, dir_rng = jax.random.split(rng)
    raw = jax.random.normal(dir_rng, (n_directions, param_space.d))
    norms = jnp.linalg.norm(raw, axis=1, keepdims=True)
    unit_dirs = raw / norms  # (n_directions, d)

    results: dict = {"delta_levels": delta_levels}

    for cond, seed_dirs in condition_seeds.items():
        print(f"\nTransfer test: {cond} ({len(seed_dirs)} seeds)")
        seed_curves = []

        for seed_dir in seed_dirs:
            rng, seed_rng = jax.random.split(rng)
            agent = PPOAgent.load(seed_dir / "final.pkl", seed_rng)

            def policy_fn(obs, _rng):
                return agent.get_deterministic_action(obs[None]).squeeze(0)

            run_episode = _make_episode_fn(
                policy_fn, command_sequence, 0.02, default_joint_pos, n_substeps=10
            )

            inject_batch = jax.jit(
                jax.vmap(lambda p: param_space.inject(base_mjx_model, p))
            )

            @jax.jit
            def eval_batch(batched_models, ep_rngs):
                def eval_one(model, rngs_for_sample):
                    out = jax.vmap(lambda r: run_episode(model, r))(rngs_for_sample)
                    return jnp.mean(out["rms_total"])
                return jax.vmap(eval_one)(batched_models, ep_rngs)

            level_errors = []
            for delta in delta_levels:
                rng, ep_rng = jax.random.split(rng)

                # p_eval = p_star + δ * direction * param_range (element-wise)
                p_evals = p_star + delta * unit_dirs * param_range[None, :]
                p_evals = jnp.clip(p_evals, finite_lower, finite_upper)

                batched_models = inject_batch(p_evals)
                ep_rngs = jax.random.split(ep_rng, n_directions * num_episodes)
                ep_rngs = ep_rngs.reshape(n_directions, num_episodes, -1)

                sample_errors = eval_batch(batched_models, ep_rngs)
                mean_err = float(jnp.mean(sample_errors))
                level_errors.append(mean_err)
                print(f"  {seed_dir.name}  δ={delta:.2f}  rms={mean_err:.4f}")
            seed_curves.append(level_errors)

        arr = np.array(seed_curves)
        results[cond] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "seeds": arr.tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# p_true sweep — the principled sysid-mismatch eval
# ---------------------------------------------------------------------------

def run_p_true_sweep(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_star: jnp.ndarray,
    p_true: jnp.ndarray,
    checkpoints_dir: str,
    command_sequence: list[dict],
    t_levels: list[float],
    conditions_to_eval: Optional[list[str]] = None,
    num_episodes: int = 10,
    rng_seed: int = 42,
) -> dict:
    """
    Evaluate all conditions at p_eval(t) = p_star + t*(p_true - p_star).

    t=0   → p_star  (where C2 pure sysid was trained; its home turf)
    t=1   → p_true  (actual ground-truth parameters; real deployment)
    t>1   → extrapolation beyond p_true

    The direction is fixed to the actual sysid mismatch vector, so this
    directly measures how each condition handles the specific error that
    CMA-ES made — unlike the transfer test which uses random directions.
    PGDR should degrade more gracefully than C2 because its posterior
    covers the p_true direction.

    Returns:
        {
          "t_levels": [...],
          "p_star_to_p_true_mahal": float,   # Mahalanobis distance for context
          "<condition>": {"mean": [...], "std": [...], "seeds": [[...], ...]},
        }
    """
    from pgdr._ppo import PPOAgent
    from pgdr.t1_env import _get_default_qpos

    checkpoints = Path(checkpoints_dir)
    default_joint_pos = jnp.array(_get_default_qpos(mj_model)[7:])
    base_mjx_model = mjx.put_model(mj_model)

    direction = p_true - p_star  # fixed mismatch vector

    norm_lower = param_space.to_normalized_vec(param_space._lowers)
    norm_upper = param_space.to_normalized_vec(param_space._uppers)
    finite_lower = jnp.where(jnp.isfinite(norm_lower), norm_lower, p_star - 1.0)
    finite_upper = jnp.where(jnp.isfinite(norm_upper), norm_upper, p_star + 1.0)

    # Group checkpoint dirs by condition (strip _seedN suffix)
    condition_seeds: dict[str, list[Path]] = {}
    for d in sorted(checkpoints.iterdir()):
        if not d.is_dir() or not (d / "final.pkl").exists():
            continue
        parts = d.name.rsplit("_seed", 1)
        cond = parts[0] if len(parts) == 2 and parts[1].isdigit() else d.name
        if conditions_to_eval and cond not in conditions_to_eval:
            continue
        condition_seeds.setdefault(cond, []).append(d)

    results: dict = {"t_levels": t_levels}

    for cond, seed_dirs in condition_seeds.items():
        print(f"\np_true sweep: {cond} ({len(seed_dirs)} seeds)")
        seed_curves = []

        for seed_dir in seed_dirs:
            rng = jax.random.PRNGKey(rng_seed)
            agent = PPOAgent.load(seed_dir / "final.pkl", rng)

            def policy_fn(obs, _rng):
                return agent.get_deterministic_action(obs[None]).squeeze(0)

            level_errors = []
            for t in t_levels:
                p_eval = jnp.clip(p_star + t * direction, finite_lower, finite_upper)
                eval_model = param_space.inject(base_mjx_model, p_eval)

                rng, ep_rng = jax.random.split(rng)
                vel = evaluate_velocity_tracking(
                    policy_fn=policy_fn,
                    mjx_model=eval_model,
                    command_sequence=command_sequence,
                    control_dt=0.02,
                    default_joint_pos=default_joint_pos,
                    num_episodes=num_episodes,
                    rng_seed=int(ep_rng[0]),
                )
                level_errors.append(vel["rms_total"])
                print(f"  {seed_dir.name}  t={t:.2f}  rms={vel['rms_total']:.4f}")
            seed_curves.append(level_errors)

        arr = np.array(seed_curves)
        results[cond] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "seeds": arr.tolist(),
        }

    return results


# ---------------------------------------------------------------------------
# Payload robustness sweep
# ---------------------------------------------------------------------------

def run_payload_sweep(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_star: jnp.ndarray,
    checkpoints_dir: str,
    command_sequence: list[dict],
    payload_levels: list[float],
    conditions_to_eval: Optional[list[str]] = None,
    num_episodes: int = 10,
) -> dict:
    """
    Sweep payload perturbation magnitude and record RMS velocity error.

    For each payload level, evaluates all matching policies and aggregates
    mean/std across seeds.

    Returns:
        {
          "payload_levels": [...],
          "<condition>": {"mean": [...], "std": [...], "seeds": [[...], ...]},
          ...
        }
    """
    from pgdr._ppo import PPOAgent
    from pgdr.t1_env import _get_default_qpos

    checkpoints = Path(checkpoints_dir)
    default_joint_pos = jnp.array(_get_default_qpos(mj_model)[7:])
    base_mjx_model = mjx.put_model(mj_model)

    # Find torso body index once
    torso_id = 1
    for i in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and "torso" in name.lower():
            torso_id = i
            break

    # Group checkpoint dirs by condition (strip _seedN)
    condition_seeds: dict[str, list[Path]] = {}
    for d in sorted(checkpoints.iterdir()):
        if not d.is_dir() or not (d / "final.pkl").exists():
            continue
        parts = d.name.rsplit("_seed", 1)
        cond = parts[0] if len(parts) == 2 and parts[1].isdigit() else d.name
        if conditions_to_eval and cond not in conditions_to_eval:
            continue
        condition_seeds.setdefault(cond, []).append(d)

    results: dict = {"payload_levels": payload_levels}

    for cond, seed_dirs in condition_seeds.items():
        print(f"\nSweeping: {cond} ({len(seed_dirs)} seeds)")
        seed_curves = []  # [n_seeds x n_levels]

        for seed_dir in seed_dirs:
            rng = jax.random.PRNGKey(0)
            agent = PPOAgent.load(seed_dir / "final.pkl", rng)

            def policy_fn(obs, _rng):
                return agent.get_deterministic_action(obs[None]).squeeze(0)

            level_errors = []
            for payload in payload_levels:
                perturbed = base_mjx_model
                if payload > 0:
                    new_mass = base_mjx_model.body_mass.at[torso_id].add(payload)
                    perturbed = base_mjx_model.replace(body_mass=new_mass)
                eval_model = param_space.inject(perturbed, p_star)
                vel = evaluate_velocity_tracking(
                    policy_fn=policy_fn,
                    mjx_model=eval_model,
                    command_sequence=command_sequence,
                    control_dt=0.02,
                    default_joint_pos=default_joint_pos,
                    num_episodes=num_episodes,
                )
                level_errors.append(vel["rms_total"])
                print(f"  {seed_dir.name}  payload={payload:.1f}kg  "
                      f"rms={vel['rms_total']:.4f}")
            seed_curves.append(level_errors)

        arr = np.array(seed_curves)  # [n_seeds, n_levels]
        results[cond] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "seeds": arr.tolist(),
        }

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
        mj_model = load_mj_model(args.model_xml)
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
        mj_model = load_mj_model(args.model_xml)
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
        )

        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nSaved results to {out}")

    else:
        parser.print_help()

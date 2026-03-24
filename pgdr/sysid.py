"""
CMA-ES System Identification with Covariance Extraction.

Core of the PGDR pipeline.  Runs CMA-ES over massively parallel MJX rollouts
to find:
    p*  = parameter vector that best reproduces reference trajectories
    Σ   = σ_K² · C_K  = covariance encoding identification uncertainty

The covariance is the key deliverable: it tells us which parameters the
optimizer could pin down (small variance) and which remained ambiguous
(large variance).  PGDR uses this to shape the DR distribution.

Supports two modes:
    - Sim-to-sim:  Reference from a known "Sim A" (ground truth available).
    - Sim-to-real: Reference from physical robot sensors.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np
import yaml

from pgdr.param_space import ParamSpace, build_t1_param_space, inject_contact_params_to_all_feet


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SysIdConfig:
    """CMA-ES identification hyperparameters."""
    popsize: int = 512
    num_generations: int = 300
    sigma_init: float = 0.3
    seed: int = 42
    w_q: float = 1.0
    w_qdot: float = 0.1
    convergence_patience: int = 30
    convergence_tol: float = 1e-6
    cov_method: str = "optimizer"       # "optimizer" or "empirical"
    empirical_top_k_frac: float = 0.25
    regularization_eps: float = 1e-6

    @classmethod
    def from_yaml(cls, path: str) -> SysIdConfig:
        with open(path) as f:
            cfg = yaml.safe_load(f)
        cmaes = cfg.get("cmaes", {})
        loss = cfg.get("loss", {})
        conv = cfg.get("convergence", {})
        cov = cfg.get("covariance", {})
        return cls(
            popsize=cmaes.get("popsize", 512),
            num_generations=cmaes.get("num_generations", 300),
            sigma_init=cmaes.get("sigma_init", 0.3),
            seed=cmaes.get("seed", 42),
            w_q=loss.get("w_q", 1.0),
            w_qdot=loss.get("w_qdot", 0.1),
            convergence_patience=conv.get("patience", 30),
            convergence_tol=conv.get("loss_tol", 1e-6),
            cov_method=cov.get("method", "optimizer"),
            empirical_top_k_frac=cov.get("empirical_top_k_frac", 0.25),
            regularization_eps=cov.get("regularization_eps", 1e-6),
        )


# ---------------------------------------------------------------------------
# Reference trajectory data
# ---------------------------------------------------------------------------

@dataclass
class ReferenceTrajectory:
    """Collected reference trajectory for identification."""
    q: jnp.ndarray       # [T, nq] joint positions
    qdot: jnp.ndarray    # [T, nv] joint velocities
    actions: jnp.ndarray # [T, nu] applied actions
    dt: float            # Control timestep

    def save(self, path: str) -> None:
        np.savez(path, q=np.array(self.q), qdot=np.array(self.qdot),
                 actions=np.array(self.actions), dt=self.dt)

    @classmethod
    def load(cls, path: str) -> ReferenceTrajectory:
        data = np.load(path)
        return cls(
            q=jnp.array(data["q"]),
            qdot=jnp.array(data["qdot"]),
            actions=jnp.array(data["actions"]),
            dt=float(data["dt"]),
        )


# ---------------------------------------------------------------------------
# Sim A creation (sim-to-sim validation)
# ---------------------------------------------------------------------------

def create_sim_a(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    perturbation_scale: float = 0.15,
    seed: int = 0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create "Sim A" by perturbing the default T1 model.

    Sim A serves as ground truth for sim-to-sim validation.
    Each parameter is perturbed by a random amount drawn from
    U[-perturbation_scale, +perturbation_scale] in normalized space.

    Args:
        mj_model:            Default MuJoCo model.
        param_space:         Parameter space definition.
        perturbation_scale:  Max perturbation in normalized units.
        seed:                Random seed.

    Returns:
        (p_true_normalized, p_true_physical):
            Ground truth parameter vector in both spaces.
    """
    rng = jax.random.PRNGKey(seed)
    d = param_space.d

    # Random perturbation in normalized space
    p_true_norm = jax.random.uniform(
        rng, shape=(d,), minval=-perturbation_scale, maxval=perturbation_scale
    )

    p_true_phys = param_space.to_physical_vec(p_true_norm)
    return p_true_norm, p_true_phys


def collect_reference_trajectory(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_normalized: jnp.ndarray,
    actions: jnp.ndarray,
    n_substeps: int = 10,
    foot_geom_ids: Optional[list[int]] = None,
) -> ReferenceTrajectory:
    """
    Roll out the simulation with given parameters and action sequence,
    recording joint positions and velocities as the reference trajectory.

    Args:
        mj_model:       Base MuJoCo model.
        param_space:    Parameter space.
        p_normalized:   Parameter vector to use (normalized).
        actions:        [T, nu] action sequence.
        n_substeps:     Physics substeps per control step.
        foot_geom_ids:  Foot geom IDs for contact param propagation.

    Returns:
        ReferenceTrajectory with recorded q, qdot, actions.
    """
    mjx_model = mjx.put_model(mj_model)
    mjx_model = param_space.inject(mjx_model, p_normalized)
    if foot_geom_ids:
        mjx_model = inject_contact_params_to_all_feet(
            mjx_model, foot_geom_ids, param_space, p_normalized
        )

    mjx_data = mjx.put_data(mj_model, mujoco.MjData(mj_model))

    # Rollout
    def step_fn(data, action):
        data = data.replace(ctrl=action)
        def substep(d, _):
            return mjx.step(mjx_model, d), None
        data, _ = jax.lax.scan(substep, data, None, length=n_substeps)
        return data, jnp.concatenate([data.qpos, data.qvel])

    _, trajectory = jax.lax.scan(step_fn, mjx_data, actions)

    nq = mj_model.nq
    q = trajectory[:, :nq]
    qdot = trajectory[:, nq:]
    dt = float(mj_model.opt.timestep * n_substeps)

    return ReferenceTrajectory(q=q, qdot=qdot, actions=actions, dt=dt)


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def _single_candidate_loss(
    p_normalized: jnp.ndarray,
    default_model: mjx.Model,
    default_data: mjx.Data,
    param_space: ParamSpace,
    ref: ReferenceTrajectory,
    n_substeps: int,
    w_q: float,
    w_qdot: float,
) -> float:
    """
    Loss for a single candidate parameter vector.

    Injects params into model, rolls out with the same actions as the
    reference trajectory, and computes MSE on q and qdot.

    This is Eq. (1) of the proposal:
        L(p) = (1/T) Σ_t [ w_q ||q_sim(p) - q_ref||² + w_qdot ||qdot_sim - qdot_ref||² ]
    """
    model = param_space.inject(default_model, p_normalized)

    def step_fn(data, action):
        data = data.replace(ctrl=action)
        def substep(d, _):
            return mjx.step(model, d), None
        data, _ = jax.lax.scan(substep, data, None, length=n_substeps)
        return data, jnp.concatenate([data.qpos, data.qvel])

    _, trajectory = jax.lax.scan(step_fn, default_data, ref.actions)

    nq = default_model.nq
    q_sim = trajectory[:, :nq]
    qdot_sim = trajectory[:, nq:]

    loss = (
        w_q * jnp.mean((q_sim - ref.q) ** 2) +
        w_qdot * jnp.mean((qdot_sim - ref.qdot) ** 2)
    )
    return loss


def batch_evaluate(
    candidates: jnp.ndarray,
    default_model: mjx.Model,
    default_data: mjx.Data,
    param_space: ParamSpace,
    ref: ReferenceTrajectory,
    n_substeps: int,
    w_q: float,
    w_qdot: float,
) -> jnp.ndarray:
    """
    Evaluate all candidate parameter vectors in parallel via jax.vmap.

    Args:
        candidates: [popsize, d] array of normalized parameter vectors.
        Other args: see _single_candidate_loss.

    Returns:
        [popsize] array of loss values.
    """
    eval_fn = jax.vmap(
        lambda p: _single_candidate_loss(
            p, default_model, default_data, param_space,
            ref, n_substeps, w_q, w_qdot,
        )
    )
    return eval_fn(candidates)


# ---------------------------------------------------------------------------
# CMA-ES identification loop
# ---------------------------------------------------------------------------

def run_identification(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    ref: ReferenceTrajectory,
    config: SysIdConfig,
    n_substeps: int = 10,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Main identification loop.

    Runs CMA-ES to find p* and extract Σ.

    Args:
        mj_model:     Default MuJoCo model.
        param_space:  (Possibly reduced) parameter space.
        ref:          Reference trajectory to match.
        config:       CMA-ES hyperparameters.
        n_substeps:   Physics substeps per control step.

    Returns:
        p_star:  [d] identified parameter vector (normalized).
        Sigma:   [d, d] covariance matrix (normalized space).
        info:    Dictionary with convergence history and diagnostics.
    """
    d = param_space.d
    rng = jax.random.PRNGKey(config.seed)

    mjx_model = mjx.put_model(mj_model)
    mjx_data = mjx.put_data(mj_model, mujoco.MjData(mj_model))

    # JIT the batch evaluation
    @jax.jit
    def evaluate_batch(candidates):
        return batch_evaluate(
            candidates, mjx_model, mjx_data, param_space,
            ref, n_substeps, config.w_q, config.w_qdot,
        )

    # --- Try evosax first ---
    try:
        import evosax
        return _run_with_evosax(
            evaluate_batch, d, config, rng,
        )
    except ImportError:
        pass

    # --- Fallback: cmaes Python package ---
    try:
        import cmaes as cmaes_pkg
        return _run_with_cmaes_pkg(
            evaluate_batch, d, config,
        )
    except ImportError:
        pass

    raise ImportError(
        "Neither evosax nor cmaes package found. Install one:\n"
        "  pip install evosax   (JAX-native, preferred)\n"
        "  pip install cmaes    (pure Python fallback)"
    )


def _run_with_evosax(
    evaluate_fn,
    d: int,
    config: SysIdConfig,
    rng: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """Run identification using evosax CMA-ES."""
    import evosax

    strategy = evosax.CMA_ES(popsize=config.popsize, num_dims=d)
    es_params = strategy.default_params
    state = strategy.initialize(rng, es_params)

    history = {"generation": [], "best_loss": [], "mean_loss": [], "sigma": []}
    best_loss = float("inf")
    patience_counter = 0
    best_mean = None
    final_candidates = None
    final_losses = None

    t0 = time.time()

    for gen in range(config.num_generations):
        rng, rng_ask = jax.random.split(rng)
        candidates, state = strategy.ask(rng_ask, state, es_params)

        losses = evaluate_fn(candidates)

        state = strategy.tell(candidates, losses, state, es_params)

        gen_best = float(jnp.min(losses))
        gen_mean = float(jnp.mean(losses))

        # Extract sigma from state (evosax stores it as sigma)
        sigma_val = float(state.sigma) if hasattr(state, "sigma") else 0.0

        history["generation"].append(gen)
        history["best_loss"].append(gen_best)
        history["mean_loss"].append(gen_mean)
        history["sigma"].append(sigma_val)

        # Convergence check
        if gen_best < best_loss - config.convergence_tol:
            best_loss = gen_best
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.convergence_patience:
            print(f"  Converged at generation {gen} (patience exhausted)")
            break

        if gen % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Gen {gen:4d}: best={gen_best:.6f}  mean={gen_mean:.6f}  "
                  f"sigma={sigma_val:.4f}  [{elapsed:.1f}s]")

        # Keep last generation for empirical covariance
        final_candidates = candidates
        final_losses = losses

    # --- Extract p* and Σ ---
    # evosax stores the mean as 'mean' or 'best_member' depending on version
    p_star = getattr(state, "mean", None)
    if p_star is None:
        p_star = getattr(state, "best_member", None)
    if p_star is None:
        # Last resort: use best candidate from final generation
        if final_candidates is not None and final_losses is not None:
            p_star = final_candidates[jnp.argmin(final_losses)]
        else:
            raise RuntimeError("Could not extract mean from evosax state")
    p_star = jnp.asarray(p_star)

    # Try to get the full covariance from evosax state
    Sigma = _extract_evosax_covariance(state, config)

    # If that failed, use empirical covariance
    if Sigma is None and final_candidates is not None:
        Sigma = _empirical_covariance(
            final_candidates, final_losses, config.empirical_top_k_frac
        )

    # Regularize
    Sigma = Sigma + config.regularization_eps * jnp.eye(d)

    elapsed = time.time() - t0
    info = {
        "history": history,
        "num_generations_run": len(history["generation"]),
        "final_loss": float(best_loss),
        "elapsed_seconds": elapsed,
        "method": "evosax",
        "cov_extraction": config.cov_method,
    }

    print(f"  Identification complete: loss={best_loss:.6f}, "
          f"{len(history['generation'])} generations, {elapsed:.1f}s")

    return p_star, Sigma, info


def _extract_evosax_covariance(state, config: SysIdConfig) -> Optional[jnp.ndarray]:
    """
    Attempt to extract σ²C from the evosax CMA-ES state.

    evosax stores the covariance decomposition differently across versions.
    We try several known field names.
    """
    if config.cov_method == "empirical":
        return None

    sigma = getattr(state, "sigma", None)
    if sigma is None:
        return None

    sigma_sq = float(sigma) ** 2

    # evosax may store C as 'C', 'cov', or 'B' and 'D' (eigendecomposition)
    C = getattr(state, "C", None)
    if C is not None and C.ndim == 2:
        return sigma_sq * C

    # Try eigendecomposition: C = B @ diag(D²) @ B^T
    B = getattr(state, "B", None)
    D = getattr(state, "D", None)
    if B is not None and D is not None:
        if D.ndim == 1:
            C = B @ jnp.diag(D ** 2) @ B.T
        else:
            C = B @ (D ** 2) @ B.T
        return sigma_sq * C

    # Some evosax versions store p_sigma and p_c but not C directly.
    # In that case, fall back to empirical.
    print("  Warning: Could not extract full covariance from evosax state. "
          "Falling back to empirical covariance.")
    return None


def _run_with_cmaes_pkg(
    evaluate_fn,
    d: int,
    config: SysIdConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, dict]:
    """
    Run identification using the `cmaes` Python package.

    This package exposes the full covariance via optimizer._C,
    making extraction reliable. The optimization loop itself runs
    on CPU, but evaluation is still JAX-parallelized on GPU.
    """
    from cmaes import CMA

    optimizer = CMA(
        mean=np.zeros(d),
        sigma=config.sigma_init,
        population_size=config.popsize,
        seed=config.seed,
    )

    history = {"generation": [], "best_loss": [], "mean_loss": [], "sigma": []}
    best_loss = float("inf")
    patience_counter = 0
    t0 = time.time()

    for gen in range(config.num_generations):
        # Ask: sample candidates (numpy)
        solutions = []
        for _ in range(config.popsize):
            x = optimizer.ask()
            solutions.append(x)

        candidates_np = np.array(solutions)
        candidates_jax = jnp.array(candidates_np)

        # Evaluate in parallel on GPU
        losses_jax = evaluate_fn(candidates_jax)
        losses_np = np.array(losses_jax)

        # Tell: update CMA-ES
        tell_list = [(solutions[i], losses_np[i]) for i in range(config.popsize)]
        optimizer.tell(tell_list)

        gen_best = float(np.min(losses_np))
        gen_mean = float(np.mean(losses_np))
        sigma_val = float(optimizer.sigma) if hasattr(optimizer, 'sigma') else 0.0

        history["generation"].append(gen)
        history["best_loss"].append(gen_best)
        history["mean_loss"].append(gen_mean)
        history["sigma"].append(sigma_val)

        if gen_best < best_loss - config.convergence_tol:
            best_loss = gen_best
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.convergence_patience:
            print(f"  Converged at generation {gen} (patience exhausted)")
            break

        if gen % 25 == 0:
            elapsed = time.time() - t0
            print(f"  Gen {gen:4d}: best={gen_best:.6f}  mean={gen_mean:.6f}  "
                  f"sigma={sigma_val:.4f}  [{elapsed:.1f}s]")

    # --- Extract p* and Σ ---
    # The cmaes package stores the mean as optimizer._mean and covariance as optimizer._C
    p_star = jnp.array(optimizer._mean)

    # Full covariance: σ² * C
    sigma_sq = float(optimizer.sigma) ** 2
    C = np.array(optimizer._C)
    Sigma = jnp.array(sigma_sq * C)
    Sigma = Sigma + config.regularization_eps * jnp.eye(d)

    elapsed = time.time() - t0
    info = {
        "history": history,
        "num_generations_run": len(history["generation"]),
        "final_loss": float(best_loss),
        "elapsed_seconds": elapsed,
        "method": "cmaes_pkg",
        "cov_extraction": "optimizer_direct",
    }

    print(f"  Identification complete: loss={best_loss:.6f}, "
          f"{len(history['generation'])} generations, {elapsed:.1f}s")

    return p_star, Sigma, info


def _empirical_covariance(
    candidates: jnp.ndarray,
    losses: jnp.ndarray,
    top_k_frac: float,
) -> jnp.ndarray:
    """
    Compute empirical covariance from the top-K candidates.

    This is a robust fallback when the optimizer's internal covariance
    is inaccessible.

    Args:
        candidates: [popsize, d] candidate vectors.
        losses:     [popsize] loss values.
        top_k_frac: Fraction of popsize to use (e.g., 0.25 = top 25%).

    Returns:
        [d, d] empirical covariance matrix.
    """
    K = max(2, int(len(losses) * top_k_frac))
    top_indices = jnp.argsort(losses)[:K]
    top_candidates = candidates[top_indices]

    # Covariance of the top performers
    mean = jnp.mean(top_candidates, axis=0)
    centered = top_candidates - mean
    Sigma = (centered.T @ centered) / (K - 1)

    return Sigma


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def _generate_action_sequence(
    mj_model: mujoco.MjModel,
    commands_config: list[dict],
    control_dt: float,
) -> jnp.ndarray:
    """
    Generate an open-loop action sequence from scripted velocity commands.

    For sim-to-sim validation, we need a fixed action sequence that exercises
    diverse walking gaits. This generates sinusoidal joint targets that
    approximate the given velocity commands.

    In practice, you'd use a pretrained policy to generate actions from
    commands. This is a simplified version for initial testing.
    """
    nu = mj_model.nu
    actions_list = []

    for cmd in commands_config:
        duration = cmd["duration"]
        n_steps = int(duration / control_dt)
        t = jnp.linspace(0, duration, n_steps)

        # Simple sinusoidal action pattern scaled by velocity
        speed = abs(cmd.get("vx", 0)) + abs(cmd.get("vy", 0)) + abs(cmd.get("wz", 0))
        freq = 2.0 + speed  # Faster gait at higher speed
        amplitude = 0.2 * max(speed, 0.1)

        # Distribute across joints with phase offsets
        phases = jnp.linspace(0, 2 * jnp.pi, nu, endpoint=False)
        action_block = amplitude * jnp.sin(freq * t[:, None] + phases[None, :])
        actions_list.append(action_block)

    return jnp.concatenate(actions_list, axis=0)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CMA-ES System Identification")
    subparsers = parser.add_subparsers(dest="command")

    # --- create-sim-a ---
    p_sim_a = subparsers.add_parser("create-sim-a",
                                     help="Create Sim A ground truth")
    p_sim_a.add_argument("--model-xml", type=str, required=True)
    p_sim_a.add_argument("--perturbation", type=float, default=0.15)
    p_sim_a.add_argument("--output", type=str,
                         default="pgdr/data/sim_a_params.npy")

    # --- collect-reference ---
    p_ref = subparsers.add_parser("collect-reference",
                                   help="Collect reference trajectory from Sim A")
    p_ref.add_argument("--model-xml", type=str, required=True)
    p_ref.add_argument("--sim-a-params", type=str, required=True)
    p_ref.add_argument("--config", type=str, default="pgdr/config/sysid_config.yaml")
    p_ref.add_argument("--output", type=str,
                       default="pgdr/data/sim_a_reference.npz")

    # --- identify ---
    p_id = subparsers.add_parser("identify", help="Run CMA-ES identification")
    p_id.add_argument("--model-xml", type=str, required=True)
    p_id.add_argument("--reference", type=str, required=True)
    p_id.add_argument("--param-space", type=str, default=None,
                      help="Path to reduced param space JSON. "
                           "If omitted, uses full param space.")
    p_id.add_argument("--config", type=str, default="pgdr/config/sysid_config.yaml")
    p_id.add_argument("--output-p-star", type=str,
                      default="pgdr/results/p_star.npy")
    p_id.add_argument("--output-sigma", type=str,
                      default="pgdr/results/Sigma.npy")

    args = parser.parse_args()

    if args.command == "create-sim-a":
        mj_model = mujoco.MjModel.from_xml_path(args.model_xml)
        ps = build_t1_param_space(mj_model)
        p_true_norm, p_true_phys = create_sim_a(
            mj_model, ps, perturbation_scale=args.perturbation
        )
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        jnp.save(str(out), p_true_norm)
        print(f"Saved Sim A ground truth ({ps.d} params) to {out}")

    elif args.command == "collect-reference":
        mj_model = mujoco.MjModel.from_xml_path(args.model_xml)
        ps = build_t1_param_space(mj_model)
        p_true_norm = jnp.load(args.sim_a_params)

        cfg = SysIdConfig.from_yaml(args.config)

        # Load command sequence from config
        with open(args.config) as f:
            raw_cfg = yaml.safe_load(f)
        commands = raw_cfg.get("reference", {}).get("commands", [
            {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
            {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
            {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
            {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
        ])
        control_dt = raw_cfg.get("reference", {}).get("control_dt", 0.02)
        actions = _generate_action_sequence(mj_model, commands, control_dt)

        ref = collect_reference_trajectory(mj_model, ps, p_true_norm, actions)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        ref.save(str(out))
        print(f"Saved reference trajectory ({len(ref.actions)} steps) to {out}")

    elif args.command == "identify":
        mj_model = mujoco.MjModel.from_xml_path(args.model_xml)

        if args.param_space:
            ps = ParamSpace.load(args.param_space)
        else:
            ps = build_t1_param_space(mj_model)

        ref = ReferenceTrajectory.load(args.reference)
        cfg = SysIdConfig.from_yaml(args.config)

        print(f"Running identification: d={ps.d}, popsize={cfg.popsize}, "
              f"max_gen={cfg.num_generations}")
        p_star, Sigma, info = run_identification(mj_model, ps, ref, cfg)

        # Save results
        for path in [args.output_p_star, args.output_sigma]:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        jnp.save(args.output_p_star, p_star)
        jnp.save(args.output_sigma, Sigma)

        print(f"Saved p* to {args.output_p_star}")
        print(f"Saved Σ ({Sigma.shape}) to {args.output_sigma}")
        print(f"Final loss: {info['final_loss']:.6f}")
        print(f"Covariance trace: {float(jnp.trace(Sigma)):.4f}")
        print(f"Covariance rank (>1e-6): "
              f"{int(jnp.sum(jnp.linalg.eigvalsh(Sigma) > 1e-6))}/{ps.d}")

    else:
        parser.print_help()

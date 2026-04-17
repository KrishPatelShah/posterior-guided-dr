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
import mujoco_warp
import warp as wp
import numpy as np
import yaml

from pgdr.param_space import ParamSpace, build_t1_param_space, inject_contact_params_to_all_feet


# ---------------------------------------------------------------------------
# ONNX policy helpers
# ---------------------------------------------------------------------------

def _build_obs_np(
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    command: np.ndarray,
    last_action: np.ndarray,
    phase: np.ndarray,
    default_angles: np.ndarray,
) -> np.ndarray:
    """
    Build the 85-dim observation expected by the mujoco_playground T1 ONNX policy.

    Layout (matches play_t1_joystick.py OnnxController.get_obs exactly):
        local_linvel        3   — from "local_linvel" sensor
        gyro                3   — from "gyro" sensor
        gravity             3   — imu_xmat.T @ [0,0,-1]
        command             3   — [vx, vy, wz]
        joint_angles-def   23   — qpos[7:] - default_angles  (indices 0,1 zeroed)
        joint_vel          23   — qvel[6:]                    (indices 0,1 zeroed)
        last_action        23   — raw ONNX output from previous step
        phase               4   — [cos(ph0), cos(ph1), sin(ph0), sin(ph1)]
    Total: 85
    """
    linvel  = mj_data.sensor("local_linvel").data.copy()
    gyro    = mj_data.sensor("gyro").data.copy()

    imu_id  = mj_model.site("imu").id
    imu_xmat = mj_data.site_xmat[imu_id].reshape(3, 3)
    gravity  = imu_xmat.T @ np.array([0.0, 0.0, -1.0])

    joint_angles = mj_data.qpos[7:] - default_angles
    joint_vel    = mj_data.qvel[6:].copy()

    # Head joints (indices 0, 1) are not observable for locomotion
    joint_angles[:2] = 0.0
    joint_vel[:2]    = 0.0

    ph     = phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
    phase4 = np.concatenate([np.cos(ph), np.sin(ph)])

    return np.concatenate([
        linvel,          # 3
        gyro,            # 3
        gravity,         # 3
        command,         # 3
        joint_angles,    # 23
        joint_vel,       # 23
        last_action,     # 23
        phase4,          # 4
    ]).astype(np.float32)  # total: 85


def load_onnx_policy(policy_path: str) -> tuple:
    """
    Load an ONNX policy.

    Returns:
        (session, input_name, output_name)
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime not installed.  Install with:\n"
            "  pip install onnxruntime"
        )

    session     = ort.InferenceSession(str(policy_path))
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"  ONNX policy: input='{input_name}' {input_shape}  "
          f"output='{output_name}'")
    return session, input_name, output_name


def collect_reference_from_onnx_policy(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_normalized,
    onnx_session,
    onnx_input_name: str,
    onnx_output_name: str,
    commands: list,
    control_dt: float = 0.02,
    n_substeps: int = 10,
    perturb_amplitude: float = 0.0,
    perturb_freq: float = 1.0,
    warmup_steps: int = 100,
) -> "ReferenceTrajectory":
    """
    Roll out an ONNX policy on a single-world MuJoCo sim and record the
    resulting reference trajectory.

    The policy is run in closed-loop: obs → ONNX → action → step.
    Uses plain mujoco (not mujoco_warp) since only one world is needed.

    When perturb_amplitude > 0, sinusoidal perturbations with evenly-spaced
    per-joint phase offsets are added on top of the policy output after
    warmup_steps steps.  This makes joint damping identifiable from the
    velocity response without requiring the robot to walk — the same
    procedure runs safely on the real robot (policy holds balance).

    Args:
        mj_model:           Base MuJoCo model (not modified permanently).
        param_space:        Parameter space for Sim A injection.
        p_normalized:       [d] normalized Sim A parameters to inject.
        onnx_session:       ONNX InferenceSession from load_onnx_policy().
        onnx_input_name:    ONNX input node name.
        onnx_output_name:   ONNX output node name.
        commands:           List of {vx, vy, wz, duration} dicts.
        control_dt:         Control timestep (seconds).
        n_substeps:         Physics substeps per control step.
        perturb_amplitude:  Amplitude of sinusoidal joint perturbations (rad).
                            0.0 disables perturbation (default, original behaviour).
        perturb_freq:       Frequency of perturbations (Hz).
        warmup_steps:       Steps before perturbation starts (policy stabilises first).

    Returns:
        ReferenceTrajectory with recorded (q, qdot, actions).
    """
    p_norm_np = np.array(p_normalized)

    # --- Save fields we'll modify so we can restore them afterwards ---
    saved_fields: dict[str, np.ndarray] = {}
    for param in param_space.params:
        key = param.mjx_field
        if key not in saved_fields:
            saved_fields[key] = getattr(mj_model, key).copy()

    # Inject Sim A parameters in-place
    param_space.inject_cpu(mj_model, p_norm_np)

    # --- Set up MuJoCo data ---
    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] = 0.78   # approximate T1 standing height
        mj_data.qpos[3] = 1.0    # quaternion w=1 (upright)
    mujoco.mj_forward(mj_model, mj_data)

    # Policy state that persists across steps
    default_angles = (
        mj_model.key_qpos[0][7:].copy() if mj_model.nkey > 0
        else np.zeros(mj_model.nu)
    )
    last_action  = np.zeros(mj_model.nu, dtype=np.float32)
    phase        = np.array([0.0, np.pi], dtype=np.float64)
    gait_freq    = 1.5
    phase_dt     = 2.0 * np.pi * gait_freq * control_dt   # ~0.1885 rad/step

    # Per-joint phase offsets for perturbation: evenly spread across [0, 2π)
    # so each joint's sinusoid is independent → all 23 damping values observable.
    perturb_phases = np.linspace(0, 2 * np.pi, mj_model.nu, endpoint=False)

    qs: list[np.ndarray]       = []
    qdots: list[np.ndarray]    = []
    actions_out: list[np.ndarray] = []
    global_step = 0

    for cmd_dict in commands:
        command = np.array([
            cmd_dict.get("vx", 0.0),
            cmd_dict.get("vy", 0.0),
            cmd_dict.get("wz", 0.0),
        ], dtype=np.float32)
        n_steps = int(cmd_dict["duration"] / control_dt)

        for _ in range(n_steps):
            obs = _build_obs_np(
                mj_model, mj_data, command, last_action, phase, default_angles
            )
            onnx_pred = onnx_session.run(
                [onnx_output_name],
                {onnx_input_name: obs[None]},
            )[0][0]

            # ctrl = raw_output + default_angles  (action_scale = 1.0)
            ctrl = onnx_pred + default_angles

            # Joint perturbation: added after policy warmup.
            # Each joint gets a sinusoid at a different phase so all 23
            # damping values are independently excited.  The policy still
            # runs closed-loop for balance — only the recorded ctrl changes.
            if perturb_amplitude > 0.0 and global_step >= warmup_steps:
                t = (global_step - warmup_steps) * control_dt
                ctrl = ctrl + perturb_amplitude * np.sin(
                    2.0 * np.pi * perturb_freq * t + perturb_phases
                )

            mj_data.ctrl[:] = ctrl
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            qs.append(mj_data.qpos.copy())
            qdots.append(mj_data.qvel.copy())
            actions_out.append(ctrl.copy())   # record applied ctrl for replay

            last_action = onnx_pred.copy()
            phase = np.fmod(phase + phase_dt + np.pi, 2.0 * np.pi) - np.pi
            global_step += 1

    # --- Restore original model fields ---
    for key, val in saved_fields.items():
        getattr(mj_model, key)[:] = val

    return ReferenceTrajectory(
        q=np.array(qs),
        qdot=np.array(qdots),
        actions=np.array(actions_out, dtype=np.float64),
        dt=control_dt,
    )


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
    sigma_max: float = 1.0              # hard cap on CMA-ES step size
    loss_clip: float = 10.0            # clip individual losses before tell()

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
            sigma_max=cmaes.get("sigma_max", 1.0),
            loss_clip=loss.get("clip", 10.0),
        )


# ---------------------------------------------------------------------------
# Reference trajectory data
# ---------------------------------------------------------------------------

@dataclass
class ReferenceTrajectory:
    """Collected reference trajectory for identification."""
    q: np.ndarray       # [T, nq] joint positions
    qdot: np.ndarray    # [T, nv] joint velocities
    actions: np.ndarray # [T, nu] applied actions
    dt: float           # Control timestep

    def save(self, path: str) -> None:
        np.savez(path, q=self.q, qdot=self.qdot, actions=self.actions, dt=self.dt)

    @classmethod
    def load(cls, path: str) -> ReferenceTrajectory:
        data = np.load(path)
        return cls(
            q=data["q"],
            qdot=data["qdot"],
            actions=data["actions"],
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
) -> tuple[np.ndarray, np.ndarray]:
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
    rng = np.random.default_rng(seed)
    d = param_space.d

    # Random perturbation in normalized space
    p_true_norm = rng.uniform(-perturbation_scale, perturbation_scale, size=(d,))

    p_true_phys = param_space.to_physical_np(p_true_norm)
    return p_true_norm, p_true_phys


def _warp_model_with_params(
    base_model: "mujoco_warp._src.types.Model",
    param_space: ParamSpace,
    candidates: np.ndarray,
) -> "mujoco_warp._src.types.Model":
    """
    Build a mujoco_warp Model with one parameter set per world.

    The warp kernel indexes model fields as model_field[worldid % field.shape[0]],
    so setting field.shape[0] == nworld gives each world its own parameters.

    Args:
        base_model:  Single-world warp Model from mujoco_warp.put_model().
        param_space: Parameter space definition.
        candidates:  [nworld, d] array of normalized parameter vectors.
    """
    nworld = len(candidates)

    for field in ["dof_damping", "body_mass", "geom_friction",
                  "geom_solref", "geom_solimp", "actuator_gainprm"]:
        wpa = getattr(base_model, field)
        base_np = wpa.numpy()                                    # (1, n) or (1, n, vec)
        tiled = np.tile(base_np, (nworld,) + (1,) * (base_np.ndim - 1))  # (nworld, ...)
        setattr(base_model, field,
                wp.array(tiled, shape=(nworld, base_np.shape[1]), dtype=wpa.dtype)
                if base_np.ndim == 2 else
                wp.array(tiled.reshape(nworld, base_np.shape[1], -1),
                         shape=(nworld, base_np.shape[1]), dtype=wpa.dtype))

    # Inject per-world physical params
    for i, p_norm in enumerate(candidates):
        p_phys = param_space.to_physical_np(p_norm)
        for j, param in enumerate(param_space.params):
            arr = getattr(base_model, param.mjx_field).numpy()
            if arr.ndim == 2:
                arr[i, param.index] = p_phys[j]
            else:
                arr[i, param.index, param.col] = p_phys[j]
            getattr(base_model, param.mjx_field).assign(arr)

    return base_model


def _rollout_warp(
    model: "mujoco_warp._src.types.Model",
    data: "mujoco_warp._src.types.Data",
    actions: np.ndarray,
    n_substeps: int,
    fix_root: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Open-loop rollout using mujoco_warp.step for all worlds in parallel.

    Args:
        model:       Warp model (nworld parameter sets).
        data:        Warp data (nworld initial states).
        actions:     [T, nu] action sequence (broadcast to all worlds), OR
                     [nworld, T, nu] per-world action sequences.
        n_substeps:  Physics substeps per control step.
        fix_root:    If True, pin the floating-base root body after each
                     control step — root qpos stays at initial pose, root
                     qvel stays zero.  Joint DOFs are unaffected.
                     Use this for sensitivity / FIM rollouts where you want
                     clean joint-level signals without the robot drifting.

    Returns:
        q    [nworld, T, nq]
        qdot [nworld, T, nv]
    """
    nworld = data.qpos.shape[0]
    nq = data.qpos.shape[1]
    nv = data.qvel.shape[1]

    per_world = actions.ndim == 3           # [nworld, T, nu] vs [T, nu]
    T = actions.shape[-2] if per_world else len(actions)

    q_traj    = np.zeros((nworld, T, nq))
    qdot_traj = np.zeros((nworld, T, nv))

    # Snapshot initial root pose (first 7 qpos entries: 3 pos + 4 quat)
    root_qpos0 = data.qpos.numpy()[:, :7].copy() if fix_root else None

    ctrl_buf = data.ctrl.numpy()
    for t in range(T):
        if per_world:
            ctrl_buf[:] = actions[:, t, :]  # [nworld, nu] — per-world control
        else:
            ctrl_buf[:] = actions[t][None, :]  # broadcast [nu] → [nworld, nu]
        data.ctrl.assign(ctrl_buf)
        for _ in range(n_substeps):
            mujoco_warp.step(model, data)

        if fix_root:
            qpos = data.qpos.numpy()
            qvel = data.qvel.numpy()
            qpos[:, :7] = root_qpos0            # restore root position/orientation
            qvel[:, :6] = 0.0                   # zero root linear/angular velocity
            data.qpos.assign(qpos)
            data.qvel.assign(qvel)

        q_traj[:, t, :]    = data.qpos.numpy()
        qdot_traj[:, t, :] = data.qvel.numpy()

    return q_traj, qdot_traj


def collect_reference_trajectory(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    p_normalized,
    actions,
    n_substeps: int = 10,
    foot_geom_ids: Optional[list[int]] = None,
    fix_root: bool = True,
) -> ReferenceTrajectory:
    """
    Roll out the simulation with given parameters, using mujoco_warp (CPU).
    Returns a ReferenceTrajectory with recorded q, qdot, actions.
    """
    wp.init()
    actions_np = np.array(actions)
    p_norm_np  = np.array(p_normalized).reshape(1, -1)   # (1, d) — single world

    warp_model = mujoco_warp.put_model(mj_model)
    warp_model = _warp_model_with_params(warp_model, param_space, p_norm_np)
    warp_data  = mujoco_warp.make_data(mj_model, nworld=1, njmax=256, nconmax=128)

    q, qdot = _rollout_warp(warp_model, warp_data, actions_np, n_substeps, fix_root=fix_root)
    # q shape: (1, T, nq) → squeeze world dim
    dt = float(mj_model.opt.timestep * n_substeps)
    return ReferenceTrajectory(
        q=q[0],
        qdot=qdot[0],
        actions=actions_np,
        dt=dt,
    )


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

def batch_evaluate_warp(
    candidates: np.ndarray,
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    ref: ReferenceTrajectory,
    n_substeps: int,
    w_q: float,
    w_qdot: float,
) -> np.ndarray:
    """
    Evaluate all candidate parameter vectors in parallel using mujoco_warp.
    Each candidate maps to one "world"; all worlds step simultaneously.

    Args:
        candidates: [popsize, d] normalized parameter vectors.

    Returns:
        [popsize] loss values.
    """
    nworld = len(candidates)
    ref_q    = np.array(ref.q)      # [T, nq]
    ref_qdot = np.array(ref.qdot)   # [T, nv]
    actions  = np.array(ref.actions)  # [T, nu]

    # Build per-world model and initial data
    warp_model = mujoco_warp.put_model(mj_model)
    warp_model = _warp_model_with_params(warp_model, param_space, candidates)
    warp_data  = mujoco_warp.make_data(mj_model, nworld=nworld, njmax=256, nconmax=128)

    q_sim, qdot_sim = _rollout_warp(warp_model, warp_data, actions, n_substeps, fix_root=True)
    # q_sim: [nworld, T, nq]

    # Match only joint angles/velocities (skip floating base: qpos[0:7], qvel[0:6]).
    # Base position reflects global trajectory drift, which is equally affected by
    # all friction params and washes out individual joint signals.
    losses = (
        w_q    * np.mean((q_sim[:, :, 7:]    - ref_q[None, :, 7:])    ** 2, axis=(1, 2)) +
        w_qdot * np.mean((qdot_sim[:, :, 6:] - ref_qdot[None, :, 6:]) ** 2, axis=(1, 2))
    )
    losses = np.where(np.isfinite(losses), losses, 1e6)
    return losses


# ---------------------------------------------------------------------------
# CMA-ES identification loop
# ---------------------------------------------------------------------------

def run_identification(
    mj_model: mujoco.MjModel,
    param_space: ParamSpace,
    ref: ReferenceTrajectory,
    config: SysIdConfig,
    n_substeps: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Main identification loop: CMA-ES + mujoco_warp parallel rollouts.

    Each CMA-ES generation evaluates `popsize` candidates simultaneously
    by running one world per candidate inside mujoco_warp.

    Returns p_star [d], Sigma [d, d], info dict.
    """
    from cmaes import CMA

    wp.init()
    d = param_space.d

    def evaluate_batch(candidates: np.ndarray) -> np.ndarray:
        return batch_evaluate_warp(
            candidates, mj_model, param_space, ref,
            n_substeps, config.w_q, config.w_qdot,
        )

    return _run_with_cmaes_warp(evaluate_batch, d, config)


def _run_with_cmaes_warp(
    evaluate_fn,
    d: int,
    config: SysIdConfig,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """CMA-ES loop driven by the `cmaes` package, evaluation via mujoco_warp."""
    from cmaes import CMA

    # Bounds in normalized space: ±3 keeps params within physical plausibility.
    # Without bounds, sigma grows when the landscape is flat, sending candidates
    # to extreme values that cause NaN simulation explosions.
    bounds = np.array([[-3.0, 3.0]] * d)
    optimizer = CMA(
        mean=np.zeros(d),
        sigma=config.sigma_init,
        population_size=config.popsize,
        seed=config.seed,
        bounds=bounds,
    )

    history = {"generation": [], "best_loss": [], "mean_loss": [], "sigma": []}
    best_loss = float("inf")
    patience_counter = 0
    t0 = time.time()

    for gen in range(config.num_generations):
        solutions = [optimizer.ask() for _ in range(config.popsize)]
        candidates = np.array(solutions)

        losses = evaluate_fn(candidates)

        # Clip losses before telling CMA-ES so blown-up simulations don't
        # distort the covariance update toward useless regions.
        losses_clipped = np.clip(losses, 0.0, config.loss_clip)
        optimizer.tell([(solutions[i], float(losses_clipped[i])) for i in range(config.popsize)])

        # Hard cap on sigma — prevents unbounded growth when landscape is flat.
        if optimizer._sigma > config.sigma_max:
            optimizer._sigma = config.sigma_max

        gen_best = float(np.min(losses))
        gen_mean = float(np.mean(losses))
        sigma_val = float(optimizer._sigma)

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

        elapsed = time.time() - t0
        print(f"  Gen {gen:4d}: best={gen_best:.6f}  mean={gen_mean:.6f}  "
              f"sigma={sigma_val:.4f}  [{elapsed:.1f}s]")

    p_star = np.array(optimizer._mean)
    sigma_sq = float(optimizer._sigma) ** 2
    Sigma = sigma_sq * np.array(optimizer._C) + config.regularization_eps * np.eye(d)

    elapsed = time.time() - t0
    info = {
        "history": history,
        "num_generations_run": len(history["generation"]),
        "final_loss": float(best_loss),
        "elapsed_seconds": elapsed,
        "method": "cmaes_warp",
    }
    print(f"  Done: loss={best_loss:.6f}, {len(history['generation'])} gens, {elapsed:.1f}s")
    return p_star, Sigma, info


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
) -> np.ndarray:
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
        t = np.linspace(0, duration, n_steps)

        # Simple sinusoidal action pattern scaled by velocity
        speed = abs(cmd.get("vx", 0)) + abs(cmd.get("vy", 0)) + abs(cmd.get("wz", 0))
        freq = 2.0 + speed  # Faster gait at higher speed
        amplitude = 0.2 * max(speed, 0.1)

        # Distribute across joints with phase offsets
        phases = np.linspace(0, 2 * np.pi, nu, endpoint=False)
        action_block = amplitude * np.sin(freq * t[:, None] + phases[None, :])
        actions_list.append(action_block)

    return np.concatenate(actions_list, axis=0)


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
        np.save(str(out), p_true_norm)
        print(f"Saved Sim A ground truth ({ps.d} params) to {out}")

    elif args.command == "collect-reference":
        mj_model = mujoco.MjModel.from_xml_path(args.model_xml)
        ps = build_t1_param_space(mj_model)
        p_true_norm = np.load(args.sim_a_params)

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
        np.save(args.output_p_star, p_star)
        np.save(args.output_sigma, Sigma)

        print(f"Saved p* to {args.output_p_star}")
        print(f"Saved Σ ({Sigma.shape}) to {args.output_sigma}")
        print(f"Final loss: {info['final_loss']:.6f}")
        print(f"Covariance trace: {float(np.trace(Sigma)):.4f}")
        print(f"Covariance rank (>1e-6): "
              f"{int(np.sum(np.linalg.eigvalsh(Sigma) > 1e-6))}/{ps.d}")

    else:
        parser.print_help()

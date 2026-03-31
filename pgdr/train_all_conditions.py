"""
Training orchestration for all experimental conditions C1–C4.

Trains locomotion policies via PPO under each domain randomization condition,
using the same hyperparameters across all conditions for fair comparison.

Conditions:
    C1: Uniform DR        — default MuJoCo Playground hand-tuned ranges
    C2: Pure sys-id       — train at p* only
    C3: Isotropic DR      — N(p*, β²I) with matched total variance
    C4: PGDR (α=0.5,1,2)  — N(p*, αΣ) — the proposed method

Compatible with TACC Lonestar6 (SLURM) — see pgdr/tacc/ for job scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import warnings

import jax
import jax.numpy as jnp
import yaml


def _sanitize_geom_types_for_mjx(mj_model) -> int:
    """Convert cylinder geoms to capsules for broader MJX collision support."""
    import mujoco

    cyl_type = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    cap_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    converted = 0
    for gid in range(mj_model.ngeom):
        if int(mj_model.geom_type[gid]) == cyl_type:
            mj_model.geom_type[gid] = cap_type
            converted += 1
    return converted


def _sanitize_dof_frictionloss_for_mjx(mj_model) -> int:
    """Zero out dof_frictionloss because MJX currently does not implement it."""
    nonzero = mj_model.dof_frictionloss != 0.0
    changed = int(nonzero.sum())
    if changed > 0:
        mj_model.dof_frictionloss[nonzero] = 0.0
    return changed


def _sanitize_integrator_for_mjx(mj_model) -> bool:
    """Switch unsupported MuJoCo integrators to Euler for MJX compatibility."""
    import mujoco

    current = int(mj_model.opt.integrator)
    euler = int(mujoco.mjtIntegrator.mjINT_EULER)
    if current != euler:
        mj_model.opt.integrator = euler
        return True
    return False


def _put_model_with_mjx_fallback(mj_model):
    """Create MJX model with retries for known MJX unsupported features."""
    from mujoco import mjx

    while True:
        try:
            return mjx.put_model(mj_model)
        except NotImplementedError as e:
            msg = str(e)
            if "collisions not implemented" in msg:
                converted = _sanitize_geom_types_for_mjx(mj_model)
                if converted <= 0:
                    raise RuntimeError(
                        "MJX rejected model collisions, and no cylinder geoms were "
                        "found to auto-convert."
                    ) from e
                warnings.warn(
                    f"MJX collision workaround applied: converted {converted} "
                    "cylinder geoms to capsules.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if "dof_frictionloss is not implemented" in msg:
                changed = _sanitize_dof_frictionloss_for_mjx(mj_model)
                if changed <= 0:
                    raise RuntimeError(
                        "MJX rejected dof_frictionloss, but no non-zero entries "
                        "were found to sanitize."
                    ) from e
                warnings.warn(
                    f"MJX frictionloss workaround applied: zeroed {changed} "
                    "dof_frictionloss entries.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            if "mjtIntegrator.mjINT_IMPLICITFAST" in msg:
                changed = _sanitize_integrator_for_mjx(mj_model)
                if not changed:
                    raise RuntimeError(
                        "MJX rejected integrator, but integrator was already Euler."
                    ) from e
                warnings.warn(
                    "MJX integrator workaround applied: switched model integrator "
                    "to mjINT_EULER.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise


def load_identification_results(results_dir: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load p* and Σ from a previous identification run."""
    p_star = jnp.load(os.path.join(results_dir, "p_star.npy"))
    Sigma = jnp.load(os.path.join(results_dir, "Sigma.npy"))
    return p_star, Sigma


def compute_isotropic_beta(Sigma: jnp.ndarray, alpha: float) -> float:
    """
    Compute β for isotropic condition C3 such that tr(β²I) = tr(αΣ).

    This ensures C3 has the same total variance as C4, so any
    performance difference is attributable purely to anisotropy.
    """
    d = Sigma.shape[0]
    total_var = jnp.trace(alpha * Sigma)
    beta = jnp.sqrt(total_var / d)
    return float(beta)


def build_condition_configs(
    train_cfg: dict,
    p_star: jnp.ndarray,
    Sigma: jnp.ndarray,
) -> dict[str, dict]:
    """
    Build per-condition training configurations.

    All conditions share the same PPO hyperparameters; only DR differs.
    """
    conditions = {}

    # C1: Uniform DR — uses Playground defaults, no custom params needed
    conditions["C1_uniform_dr"] = {
        "dr_mode": "uniform",
        "description": "Default MuJoCo Playground uniform DR",
    }

    # C2: Pure sys-id — fixed at p*, no DR
    conditions["C2_pure_sysid"] = {
        "dr_mode": "none",
        "p_star": p_star.tolist(),
        "description": "Train at identified p* only, no randomization",
    }

    # C3: Isotropic DR — N(p*, β²I) with matched trace
    # Use α=1.0 as reference for trace matching
    beta = compute_isotropic_beta(Sigma, alpha=1.0)
    conditions["C3_isotropic"] = {
        "dr_mode": "isotropic",
        "p_star": p_star.tolist(),
        "Sigma": Sigma.tolist(),
        "alpha": 1.0,
        "beta": beta,
        "description": f"Isotropic N(p*, {beta:.4f}²I), trace-matched to PGDR α=1",
    }

    # C4: PGDR at multiple α values
    for alpha in [0.5, 1.0, 2.0]:
        name = f"C4_pgdr_{alpha}"
        conditions[name] = {
            "dr_mode": "pgdr",
            "p_star": p_star.tolist(),
            "Sigma": Sigma.tolist(),
            "alpha": alpha,
            "description": f"PGDR N(p*, {alpha}Σ)",
        }

    return conditions


def train_single_condition(
    condition_name: str,
    condition_cfg: dict,
    train_cfg: dict,
    seed: int,
    checkpoint_dir: str,
    model_xml: str,
    param_space_path: str,
    dry_run: bool = False,
) -> dict:
    """
    Train a single policy under one condition and one seed.

    This function constructs and runs the training loop, integrating
    the PGDR randomizer into MuJoCo Playground's PPO pipeline.

    Args:
        condition_name:   e.g., "C4_pgdr_1.0"
        condition_cfg:    DR-specific config for this condition.
        train_cfg:        Shared PPO hyperparameters.
        seed:             Random seed.
        checkpoint_dir:   Where to save the trained policy.
        model_xml:        Path to T1 MuJoCo XML.
        param_space_path: Path to param space JSON.
        dry_run:          If True, only print config without training.

    Returns:
        Dictionary with training results (final reward, wall time, etc.).
    """
    import mujoco
    from mujoco import mjx

    from pgdr.param_space import ParamSpace, build_t1_param_space, _find_foot_geoms
    from pgdr.pgdr_randomizer import PGDRRandomizer, DRMode, build_randomizer

    run_name = f"{condition_name}_seed{seed}"
    save_dir = Path(checkpoint_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    full_cfg = {
        "condition": condition_name,
        "condition_config": condition_cfg,
        "training": train_cfg.get("training", {}),
        "seed": seed,
    }
    with open(save_dir / "config.json", "w") as f:
        json.dump(full_cfg, f, indent=2, default=str)

    if dry_run:
        print(f"  [DRY RUN] Would train {run_name}")
        return {"status": "dry_run", "condition": condition_name, "seed": seed}

    # --- Load model and parameter space ---
    mj_model = mujoco.MjModel.from_xml_path(model_xml)
    if param_space_path and Path(param_space_path).exists():
        ps = ParamSpace.load(param_space_path)
    else:
        ps = build_t1_param_space(mj_model)

    foot_geom_ids = _find_foot_geoms(mj_model)

    # --- Build randomizer ---
    p_star = jnp.array(condition_cfg["p_star"]) if "p_star" in condition_cfg else None
    Sigma = jnp.array(condition_cfg["Sigma"]) if "Sigma" in condition_cfg else None
    alpha = condition_cfg.get("alpha", 1.0)

    randomizer = build_randomizer(
        param_space=ps,
        mode=condition_cfg["dr_mode"],
        p_star=p_star,
        Sigma=Sigma,
        alpha=alpha,
        foot_geom_ids=foot_geom_ids,
    )

    print(f"  {run_name}: {randomizer.describe()}")

    # --- Training setup ---
    from pgdr._ppo import PPOAgent, PPOConfig, make_obs

    rng = jax.random.PRNGKey(seed)
    mjx_model_default = _put_model_with_mjx_fallback(mj_model)

    tcfg = train_cfg.get("training", {})
    num_envs = tcfg.get("num_envs", 4096)
    episode_length = tcfg.get("episode_length", 1000)
    total_timesteps = tcfg.get("total_timesteps", 100_000_000)
    num_steps = tcfg.get("num_steps", 10)
    control_dt = train_cfg.get("environment", {}).get("control_dt", 0.02)
    sim_dt = train_cfg.get("environment", {}).get("sim_dt", 0.002)
    action_scale = train_cfg.get("environment", {}).get("action_scale", 0.25)

    num_updates = total_timesteps // (num_envs * num_steps)

    ppo_cfg = PPOConfig(
        num_envs=num_envs,
        num_steps=num_steps,
        learning_rate=tcfg.get("learning_rate", 3e-4),
        gamma=tcfg.get("gamma", 0.99),
        gae_lambda=tcfg.get("gae_lambda", 0.95),
        clip_eps=tcfg.get("clip_eps", 0.2),
        entropy_coef=tcfg.get("entropy_coef", 0.01),
        value_coef=tcfg.get("value_coef", 0.5),
        max_grad_norm=tcfg.get("max_grad_norm", 0.5),
        num_minibatches=tcfg.get("num_minibatches", 32),
        update_epochs=tcfg.get("update_epochs", 5),
        action_scale=action_scale,
        sim_dt=sim_dt,
        control_dt=control_dt,
    )

    # Obs dim: qpos (skip x,y = nq-2) + qvel (nv) + cmd (3)
    obs_dim = (mj_model.nq - 2) + mj_model.nv + 3
    act_dim = mj_model.nu

    agent = PPOAgent(ppo_cfg, obs_dim, act_dim, rng)

    # Episode management: re-randomize params every episode_length steps
    steps_in_episode = 0
    episode_reset_interval = episode_length // num_steps

    # --- Main training loop ---
    import time as time_mod

    t0 = time_mod.time()
    best_reward = -float("inf")

    # Initial environment setup
    rng, rng_reset, rng_cmd = jax.random.split(rng, 3)
    batched_model, _ = randomizer.apply_batch(rng_reset, mjx_model_default, num_envs)
    batched_data = jax.vmap(mjx.make_data)(batched_model)

    # Sample velocity commands: [num_envs, 3] = [vx, vy, wz]
    cmd = jnp.concatenate([
        jax.random.uniform(rng_cmd, (num_envs, 1), minval=-1.0, maxval=2.0),   # vx
        jax.random.uniform(rng_cmd, (num_envs, 1), minval=-0.5, maxval=0.5),   # vy
        jax.random.uniform(rng_cmd, (num_envs, 1), minval=-1.0, maxval=1.0),   # wz
    ], axis=1)

    for update in range(num_updates):
        rng, rng_step, rng_reset, rng_cmd = jax.random.split(rng, 4)

        # Re-randomize environments periodically (new episode)
        if update % episode_reset_interval == 0 and update > 0:
            batched_model, _ = randomizer.apply_batch(rng_reset, mjx_model_default, num_envs)
            batched_data = jax.vmap(mjx.make_data)(batched_model)
            # New random commands
            cmd = jnp.concatenate([
                jax.random.uniform(rng_cmd, (num_envs, 1), minval=-1.0, maxval=2.0),
                jax.random.uniform(rng_cmd, (num_envs, 1), minval=-0.5, maxval=0.5),
                jax.random.uniform(rng_cmd, (num_envs, 1), minval=-1.0, maxval=1.0),
            ], axis=1)

        # Collect rollout
        rollout = agent.collect_rollout(batched_model, batched_data, cmd, rng_step)

        # Update env state for next rollout
        batched_data = rollout.pop("next_data")

        # PPO update
        loss_info = agent.update(rollout)

        if update % 100 == 0:
            elapsed = time_mod.time() - t0
            steps_done = (update + 1) * num_envs * num_steps
            fps = steps_done / max(elapsed, 1)
            mean_reward = loss_info.get("mean_episode_reward", 0.0)
            print(f"    Update {update}/{num_updates}: "
                  f"reward={mean_reward:.3f}  "
                  f"loss={loss_info.get('total_loss', 0):.4f}  "
                  f"entropy={loss_info.get('entropy', 0):.3f}  "
                  f"fps={fps:.0f}  [{elapsed:.0f}s]")

            if mean_reward > best_reward:
                best_reward = mean_reward
                agent.save(save_dir / "best.pkl")

        # Save checkpoint periodically
        if update % 1000 == 0 and update > 0:
            agent.save(save_dir / f"checkpoint_{update}.pkl")

    # Final save
    agent.save(save_dir / "final.pkl")
    elapsed = time_mod.time() - t0

    results = {
        "condition": condition_name,
        "seed": seed,
        "status": "completed",
        "num_updates": num_updates,
        "elapsed_seconds": elapsed,
        "best_reward": float(best_reward),
        "checkpoint_dir": str(save_dir),
    }

    with open(save_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def train_all(args):
    """Train all conditions across all seeds."""
    # Load config
    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)

    # Load identification results
    p_star, Sigma = load_identification_results(args.results_dir)
    print(f"Loaded identification results: d={p_star.shape[0]}")
    print(f"  Σ trace: {float(jnp.trace(Sigma)):.4f}")
    print(f"  Σ condition number: "
          f"{float(jnp.max(jnp.linalg.eigvalsh(Sigma)) / jnp.max(jnp.array([jnp.min(jnp.linalg.eigvalsh(Sigma)), 1e-10]))):.1f}")

    # Build condition configs
    conditions = build_condition_configs(train_cfg, p_star, Sigma)

    seeds = train_cfg.get("seeds", [0, 1, 2])
    if args.seeds:
        seeds = [int(s) for s in args.seeds]

    # Filter conditions if specified
    if args.conditions:
        conditions = {k: v for k, v in conditions.items() if k in args.conditions}

    print(f"\nTraining plan: {len(conditions)} conditions x {len(seeds)} seeds "
          f"= {len(conditions) * len(seeds)} runs")
    for name, cfg in conditions.items():
        print(f"  {name}: {cfg['description']}")

    if args.dry_run:
        print("\n[DRY RUN] No training will be executed.")
        return

    # Run training
    all_results = []
    for name, cfg in conditions.items():
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Training: {name}, seed={seed}")
            print(f"{'='*60}")

            result = train_single_condition(
                condition_name=name,
                condition_cfg=cfg,
                train_cfg=train_cfg,
                seed=seed,
                checkpoint_dir=args.checkpoint_dir,
                model_xml=args.model_xml,
                param_space_path=args.param_space,
                dry_run=args.dry_run,
            )
            all_results.append(result)

    # Save summary
    summary_path = Path(args.checkpoint_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nTraining complete. Summary saved to {summary_path}")


# ---------------------------------------------------------------------------
# SLURM launcher for TACC Lonestar6
# ---------------------------------------------------------------------------

def generate_slurm_jobs(args):
    """Generate SLURM job scripts for each condition/seed combination."""
    with open(args.config) as f:
        train_cfg = yaml.safe_load(f)

    p_star, Sigma = load_identification_results(args.results_dir)
    conditions = build_condition_configs(train_cfg, p_star, Sigma)

    seeds = train_cfg.get("seeds", [0, 1, 2])
    slurm_dir = Path(args.slurm_dir)
    slurm_dir.mkdir(parents=True, exist_ok=True)

    job_scripts = []
    for name, cfg in conditions.items():
        for seed in seeds:
            run_name = f"{name}_seed{seed}"
            script_path = slurm_dir / f"train_{run_name}.sh"

            script = f"""#!/bin/bash
#SBATCH -J pgdr_{run_name}
#SBATCH -o {slurm_dir}/logs/{run_name}_%j.out
#SBATCH -e {slurm_dir}/logs/{run_name}_%j.err
#SBATCH -p gpu-a100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A <ALLOCATION>

# TACC Lonestar6 module setup
module load gcc/11.2.0
module load cuda/12.0
module load python3/3.11

# Activate environment
source $WORK/pgdr_env/bin/activate

# Set JAX to use GPU
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$TACC_CUDA_DIR"
export JAX_PLATFORMS="gpu"

cd $WORK/SPI-Active

python -m pgdr.train_all_conditions train \\
    --model-xml {args.model_xml} \\
    --config {args.config} \\
    --results-dir {args.results_dir} \\
    --checkpoint-dir {args.checkpoint_dir} \\
    --conditions {name} \\
    --seeds {seed}
"""
            with open(script_path, "w") as f:
                f.write(script)
            job_scripts.append(script_path)

    # Generate a master submission script
    master_path = slurm_dir / "submit_all.sh"
    with open(master_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all PGDR training jobs\n\n")
        f.write(f"mkdir -p {slurm_dir}/logs\n\n")
        for sp in job_scripts:
            f.write(f"sbatch {sp}\n")
        f.write(f"\necho 'Submitted {len(job_scripts)} jobs'\n")

    os.chmod(master_path, 0o755)
    print(f"Generated {len(job_scripts)} SLURM scripts in {slurm_dir}/")
    print(f"Submit all: bash {master_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train policies under all experimental conditions"
    )
    subparsers = parser.add_subparsers(dest="command")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Run training")
    p_train.add_argument("--model-xml", type=str, required=True)
    p_train.add_argument("--config", type=str,
                         default="pgdr/config/train_config.yaml")
    p_train.add_argument("--results-dir", type=str,
                         default="pgdr/results",
                         help="Directory with p_star.npy and Sigma.npy")
    p_train.add_argument("--param-space", type=str, default=None,
                         help="Path to reduced param space JSON")
    p_train.add_argument("--checkpoint-dir", type=str,
                         default="pgdr/checkpoints")
    p_train.add_argument("--conditions", nargs="+", default=None,
                         help="Train only these conditions")
    p_train.add_argument("--seeds", nargs="+", default=None,
                         help="Train only these seeds")
    p_train.add_argument("--dry-run", action="store_true")

    # --- slurm ---
    p_slurm = subparsers.add_parser("slurm",
                                     help="Generate SLURM scripts for TACC")
    p_slurm.add_argument("--model-xml", type=str, required=True)
    p_slurm.add_argument("--config", type=str,
                         default="pgdr/config/train_config.yaml")
    p_slurm.add_argument("--results-dir", type=str, default="pgdr/results")
    p_slurm.add_argument("--checkpoint-dir", type=str,
                         default="pgdr/checkpoints")
    p_slurm.add_argument("--slurm-dir", type=str, default="pgdr/tacc/jobs")

    args = parser.parse_args()

    if args.command == "train":
        train_all(args)
    elif args.command == "slurm":
        generate_slurm_jobs(args)
    else:
        parser.print_help()

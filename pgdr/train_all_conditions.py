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
import platform
import subprocess
import sys
from pathlib import Path

if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from pgdr.model_utils import load_mj_model, resolve_model_xml, resolve_param_space_path


def load_identification_results(results_dir: str) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Load p* and Σ from a previous identification run."""
    p_star = jnp.array(np.load(os.path.join(results_dir, "p_star.npy")))
    Sigma = jnp.array(np.load(os.path.join(results_dir, "Sigma.npy")))
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
    from pgdr.param_space import ParamSpace, build_t1_param_space, _find_foot_geoms
    from pgdr.pgdr_randomizer import build_randomizer
    from mujoco import mjx

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
    mj_model = load_mj_model(model_xml)
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

    # --- Training loop ---
    # This integrates with MuJoCo Playground's PPO training.
    # The key modification: at each episode reset, we sample physical
    # parameters from the randomizer instead of using uniform DR.

    rng = jax.random.PRNGKey(seed)
    mjx_model_default = mjx.put_model(mj_model)

    tcfg = train_cfg.get("training", {})
    num_envs = tcfg.get("num_envs", 4096)
    episode_length = tcfg.get("episode_length", 1000)
    total_timesteps = tcfg.get("total_timesteps", 100_000_000)
    lr = tcfg.get("learning_rate", 3e-4)
    gamma = tcfg.get("gamma", 0.99)
    gae_lambda = tcfg.get("gae_lambda", 0.95)
    clip_eps = tcfg.get("clip_eps", 0.2)
    entropy_coef = tcfg.get("entropy_coef", 0.01)
    value_coef = tcfg.get("value_coef", 0.5)
    max_grad_norm = tcfg.get("max_grad_norm", 0.5)
    num_minibatches = tcfg.get("num_minibatches", 32)
    update_epochs = tcfg.get("update_epochs", 5)
    num_steps = tcfg.get("num_steps", 10)

    num_updates = total_timesteps // (num_envs * num_steps)

    # --- PPO agent + environment setup ---
    from pgdr._ppo import PPOAgent, PPOConfig
    from pgdr.t1_env import PGDREnv, OBS_DIM, ACT_DIM

    ppo_cfg = PPOConfig(
        num_envs=num_envs,
        num_steps=num_steps,
        learning_rate=lr,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_eps=clip_eps,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=max_grad_norm,
        num_minibatches=num_minibatches,
        update_epochs=update_epochs,
    )

    env = PGDREnv(
        mj_model=mj_model,
        randomizer=randomizer,
        max_episode_steps=episode_length,
    )

    agent = PPOAgent(ppo_cfg, OBS_DIM, ACT_DIM, rng)

    # --- Main training loop ---
    import time as time_mod

    t0 = time_mod.time()
    episode_returns = []

    # Initial env reset
    rng, reset_rng = jax.random.split(rng)
    env_state = env.reset(reset_rng, num_envs=num_envs)

    for update in range(num_updates):
        rng, rollout_rng = jax.random.split(rng)

        # Collect rollout from current env state
        rollout, env_state, rng = agent.collect_rollout(
            env, env_state, num_steps, rollout_rng
        )

        # Bootstrap value for last state
        last_value = jax.vmap(
            lambda o: agent.get_value(agent.state.params, o[None]).squeeze()
        )(env_state.obs)

        # PPO update
        loss_info, rng = agent.update(rollout, last_value, rng)

        # Auto-reset environments that finished
        rng, reset_rng2 = jax.random.split(rng)
        done_mask = env_state.done
        if jnp.any(done_mask):
            new_state = env.reset(reset_rng2, num_envs=num_envs)
            # Replace done environments with fresh resets
            def _merge(new, old, mask):
                return jnp.where(mask[:, None] if new.ndim > 1 else mask, new, old)
            env_state = env_state._replace(
                mjx_data=jax.tree_util.tree_map(
                    lambda n, o: jnp.where(
                        done_mask.reshape((-1,) + (1,) * (n.ndim - 1)), n, o
                    ),
                    new_state.mjx_data, env_state.mjx_data,
                ),
                mjx_model=jax.tree_util.tree_map(
                    lambda n, o: jnp.where(
                        done_mask.reshape((-1,) + (1,) * (n.ndim - 1)), n, o
                    ),
                    new_state.mjx_model, env_state.mjx_model,
                ),
                obs=jnp.where(done_mask[:, None], new_state.obs, env_state.obs),
                command=jnp.where(done_mask[:, None], new_state.command, env_state.command),
                step_count=jnp.where(done_mask, new_state.step_count, env_state.step_count),
                done=jnp.zeros_like(env_state.done),
            )

        if update % 100 == 0:
            elapsed = time_mod.time() - t0
            steps_done = (update + 1) * num_envs * num_steps
            fps = steps_done / max(elapsed, 1e-6)
            mean_return = float(jnp.mean(rollout.reward))
            episode_returns.append(mean_return)
            print(f"    Update {update}/{num_updates}: "
                  f"return={mean_return:.3f}  "
                  f"loss={loss_info.get('total_loss', 0):.4f}  "
                  f"fps={fps:.0f}  [{elapsed:.0f}s]")

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
        "episode_returns": [float(r) for r in episode_returns],
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
                param_space_path=resolve_param_space_path(args.param_space, args.results_dir),
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

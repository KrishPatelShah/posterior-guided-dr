#!/usr/bin/env python3
"""
run_sysid_pipeline.py — Full PGDR system identification pipeline.

Runs all four sys-id stages in order and saves everything to one
timestamped directory:

  1. create-sim-a      Perturb the T1 model to create ground truth (p_true).
  2. collect-reference Roll out Sim A to produce the reference trajectory.
  3. identify          CMA-ES over parallel rollouts → p_star, Sigma.
  4. verify            Check covariance calibration against p_true.

Outputs (all inside --output-dir):
  param_space.json          Parameter space used
  p_true.npy                Ground truth perturbation (Sim A)
  reference.npz             Reference trajectory
  p_star.npy                Identified parameters
  Sigma.npy                 Identification covariance
  sysid_info.json           Per-generation CMA-ES history
  identification_summary.json  Final loss, generations, Sigma stats
  calibration.json          Pearson/Spearman correlation + per-param errors
  figures/                  Convergence, covariance, and calibration plots

Usage:
  # Recreate the friction-only smoke results (matches 20260402_183554_friction_only)
  python run_sysid_pipeline.py --model-xml t1

  # Full parameter space, full config (for TACC)
  python run_sysid_pipeline.py --model-xml t1 --groups all --config pgdr/config/sysid_config.yaml

  # Specific groups, custom output directory
  python run_sysid_pipeline.py --model-xml t1 --groups friction mass --output-dir pgdr/results/my_run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import yaml

if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

from pgdr.model_utils import load_mj_model
from pgdr.param_space import build_t1_param_space
from pgdr.sysid import (
    SysIdConfig,
    create_sim_a,
    collect_reference_trajectory,
    run_identification,
    _generate_action_sequence,
)
from pgdr.evaluate import compute_covariance_calibration, compute_param_recovery


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PGDR sys-id pipeline (all four stages)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model-xml",
        default="t1",
        help='MuJoCo model path or alias. Use "t1" for MuJoCo Playground. (default: t1)',
    )
    parser.add_argument(
        "--config",
        default="pgdr/config/sysid_config_smoke.yaml",
        help="Sys-id config YAML. (default: sysid_config_smoke.yaml)",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["friction"],
        metavar="GROUP",
        help=(
            "Parameter groups to identify. "
            "Options: friction mass actuator contact all. "
            "(default: friction)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to pgdr/results/<timestamp>_<groups>.",
    )
    parser.add_argument(
        "--perturb",
        type=float,
        default=0.15,
        help="Sim A perturbation scale in normalized space. (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for Sim A creation. (default: 42)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Setup
    # ------------------------------------------------------------------ #
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if "all" in args.groups:
        group_tag = "full"
        filter_groups = None
    else:
        group_tag = "_".join(args.groups)
        filter_groups = args.groups

    out_dir = Path(args.output_dir or f"pgdr/results/{timestamp}_{group_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"PGDR Sys-ID Pipeline")
    print(f"{'='*60}")
    print(f"  Model:      {args.model_xml}")
    print(f"  Config:     {args.config}")
    print(f"  Groups:     {group_tag}")
    print(f"  Output dir: {out_dir}")
    print()

    mj_model = load_mj_model(args.model_xml)

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = SysIdConfig.from_yaml(args.config)

    # ------------------------------------------------------------------ #
    # Build parameter space
    # ------------------------------------------------------------------ #
    ps = build_t1_param_space(mj_model)
    if filter_groups is not None:
        ps = ps.select_by_group(filter_groups)
    ps.save(str(out_dir / "param_space.json"))

    group_counts = {g: len(ps.group_indices(g)) for g in set(p.group for p in ps.params)}
    print(f"Parameter space: d={ps.d}  {group_counts}")

    # ------------------------------------------------------------------ #
    # Stage 1: Create Sim A
    # ------------------------------------------------------------------ #
    print(f"\n[1/4] Creating Sim A (perturb={args.perturb}, seed={args.seed})...")
    p_true_norm, _ = create_sim_a(
        mj_model, ps, perturbation_scale=args.perturb, seed=args.seed
    )
    np.save(str(out_dir / "p_true.npy"), p_true_norm)
    print(f"  Saved p_true.npy  (d={p_true_norm.shape[0]})")

    # ------------------------------------------------------------------ #
    # Stage 2: Collect reference trajectory
    # ------------------------------------------------------------------ #
    print("\n[2/4] Collecting reference trajectory...")
    commands = raw_cfg.get("reference", {}).get("commands", [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
    ])
    control_dt = raw_cfg.get("reference", {}).get("control_dt", 0.02)
    actions = _generate_action_sequence(mj_model, commands, control_dt)

    ref = collect_reference_trajectory(mj_model, ps, p_true_norm, actions)
    ref.save(str(out_dir / "reference.npz"))
    print(f"  Saved reference.npz  (T={ref.q.shape[0]} steps, nu={ref.actions.shape[1]})")

    # ------------------------------------------------------------------ #
    # Stage 3: CMA-ES identification
    # ------------------------------------------------------------------ #
    print(f"\n[3/4] Running CMA-ES identification "
          f"(popsize={cfg.popsize}, max_gen={cfg.num_generations})...")
    p_star, Sigma, info = run_identification(mj_model, ps, ref, cfg)

    np.save(str(out_dir / "p_star.npy"), p_star)
    np.save(str(out_dir / "Sigma.npy"), Sigma)
    (out_dir / "sysid_info.json").write_text(json.dumps(info, indent=2, default=float))

    eigvals = np.linalg.eigvalsh(Sigma)
    summary = {
        "final_loss": float(info["final_loss"]),
        "num_generations_run": int(info["num_generations_run"]),
        "sigma_trace": float(np.trace(Sigma)),
        "sigma_condition_number": float(np.max(eigvals) / max(float(np.min(eigvals)), 1e-10)),
    }
    (out_dir / "identification_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  Saved p_star.npy, Sigma.npy")
    print(f"  Final loss:    {summary['final_loss']:.6f}")
    print(f"  Generations:   {summary['num_generations_run']}")
    print(f"  Sigma trace:   {summary['sigma_trace']:.4f}")
    print(f"  Sigma cond#:   {summary['sigma_condition_number']:.4f}")

    # ------------------------------------------------------------------ #
    # Stage 4: Verify covariance calibration
    # ------------------------------------------------------------------ #
    print("\n[4/4] Verifying covariance calibration...")
    p_star_jnp = jnp.array(p_star)
    p_true_jnp = jnp.array(p_true_norm)
    Sigma_jnp  = jnp.array(Sigma)

    recovery    = compute_param_recovery(p_star_jnp, p_true_jnp, ps)
    calibration = compute_covariance_calibration(p_star_jnp, p_true_jnp, Sigma_jnp, ps)

    (out_dir / "calibration.json").write_text(
        json.dumps({**calibration, "recovery": recovery}, indent=2)
    )
    print(f"  Param recovery RMSE:  {recovery['total_rmse']:.4f}")
    print(f"  Pearson correlation:  {calibration['pearson_correlation']:.3f}")
    print(f"  Spearman correlation: {calibration['spearman_correlation']:.3f}")
    print(f"  → {calibration['interpretation']}")

    # ------------------------------------------------------------------ #
    # Generate plots
    # ------------------------------------------------------------------ #
    print("\n[+] Generating plots...")
    try:
        from pgdr.plotting import generate_all_plots
        generate_all_plots(str(out_dir))
        print(f"  Saved to {out_dir / 'figures'}/")
    except Exception as e:
        print(f"  Warning: plotting failed ({e}). Results are still saved.")

    # ------------------------------------------------------------------ #
    # Final summary
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"Done. Results saved to: {out_dir}")
    print(f"{'='*60}")
    print(f"  p_star.npy              Identified parameters (d={ps.d})")
    print(f"  Sigma.npy               Identification covariance ({ps.d}x{ps.d})")
    print(f"  calibration.json        Pearson r={calibration['pearson_correlation']:.3f}")
    print(f"  identification_summary.json")
    print()
    print(f"Next step — policy training:")
    print(f"  python run_train.py train \\")
    print(f"      --model-xml {args.model_xml} \\")
    print(f"      --results-dir {out_dir} \\")
    print(f"      --config pgdr/config/train_config_smoke.yaml \\")
    print(f"      --checkpoint-dir pgdr/checkpoints/{out_dir.name}_smoke")


if __name__ == "__main__":
    main()

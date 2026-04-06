"""
run_sysid.py — Entry point for PGDR system identification.

Runs CMA-ES over parallel MJX rollouts to identify the T1's physical
parameters (p*) and covariance (Σ) from reference trajectories.

Typical workflow:

  # 1. Run sensitivity analysis to find which params matter most
  python run_sysid.py sensitivity --model-xml path/to/t1.xml \\
      --output pgdr/results/sensitivity.json

  # 2. Create Sim A with perturbed parameters (sim-to-sim mode)
  python run_sysid.py create-sim-a --model-xml path/to/t1.xml \\
      --output-dir pgdr/results/

  # 3. Collect reference trajectories from Sim A
  python run_sysid.py collect-reference --model-xml path/to/t1.xml \\
      --sim-a-params pgdr/results/p_true.npy \\
      --output pgdr/results/reference.npz

  # 4. Run CMA-ES identification
  python run_sysid.py identify --model-xml path/to/t1.xml \\
      --reference pgdr/results/reference.npz \\
      --output-dir pgdr/results/

  # 5. Verify covariance calibration (sim-to-sim only)
  python run_sysid.py verify --model-xml path/to/t1.xml \\
      --results-dir pgdr/results/
"""

import argparse
import json
import os
import platform
import sys
from pathlib import Path

if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from pgdr.model_utils import load_mj_model, resolve_model_xml


def cmd_sensitivity(args):
    from pgdr.param_space import build_t1_param_space
    from pgdr.sensitivity import run_sensitivity_analysis, reduce_param_space
    from pgdr.sysid import _generate_action_sequence

    mj_model = load_mj_model(args.model_xml)
    ps = build_t1_param_space(mj_model)

    print(f"Running one-at-a-time sensitivity analysis (d={ps.d})...")
    commands = [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 2.0},
    ]
    actions = _generate_action_sequence(mj_model, commands, control_dt=0.02)
    scores = run_sensitivity_analysis(
        mj_model=mj_model,
        param_space=ps,
        actions=actions[:args.horizon],
        perturbation=args.perturb,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=2))
    print(f"Sensitivity scores saved to {out}")

    ranked = sorted(
        scores["sensitivity_scores"].items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    print("\nTop 10 most sensitive parameters:")
    for name, score in ranked[:10]:
        print(f"  {name:35s}  {score:.4f}")

    if args.top_k:
        reduced_ps, _ = reduce_param_space(ps, scores, keep_top_k=args.top_k)
        reduced_path = out.with_suffix(".reduced_space.json")
        reduced_ps.save(str(reduced_path))
        print(f"\nReduced param space (top {args.top_k}) saved to {reduced_path}")


def cmd_create_sim_a(args):
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.sysid import create_sim_a

    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(args.param_space)
          if args.param_space else build_t1_param_space(mj_model))

    if args.friction_only:
        ps = ps.select_by_group(["friction"])
        print(f"Using friction-only param space (d={ps.d})")

    p_true_norm, p_true_phys = create_sim_a(
        mj_model=mj_model,
        param_space=ps,
        perturbation_scale=args.perturb,
        seed=args.seed,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_dir / "p_true.npy"), np.array(p_true_norm))
    ps.save(str(out_dir / "param_space.json"))

    print(f"Sim A created: d={p_true_norm.shape[0]}, perturb={args.perturb}")
    print(f"  p_true (normalized) saved to {out_dir / 'p_true.npy'}")


def cmd_collect_reference(args):
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.sysid import (
        collect_reference_trajectory, SysIdConfig, _generate_action_sequence,
        collect_reference_from_onnx_policy, load_onnx_policy,
    )

    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(args.param_space)
          if args.param_space else build_t1_param_space(mj_model))

    cfg_path = args.config
    _ = SysIdConfig.from_yaml(cfg_path) if cfg_path else SysIdConfig()

    # Load or zero-init parameters
    if args.sim_a_params:
        p_norm = np.load(args.sim_a_params)
        print(f"Using Sim A params from {args.sim_a_params}")
    else:
        p_norm = np.zeros(ps.d)
        print("Using default parameters (real-robot mode — inject real data manually)")

    raw_cfg = {}
    if cfg_path:
        with open(cfg_path) as f:
            raw_cfg = yaml.safe_load(f) or {}
    commands = raw_cfg.get("reference", {}).get("commands", [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
    ])
    control_dt = raw_cfg.get("reference", {}).get("control_dt", 0.02)

    if args.onnx_policy:
        print(f"Loading ONNX policy from {args.onnx_policy}...")
        session, input_name, output_name = load_onnx_policy(args.onnx_policy)
        print(f"Rolling out ONNX policy (closed-loop)...")
        ref = collect_reference_from_onnx_policy(
            mj_model=mj_model,
            param_space=ps,
            p_normalized=p_norm,
            onnx_session=session,
            onnx_input_name=input_name,
            onnx_output_name=output_name,
            commands=commands,
            control_dt=control_dt,
            n_substeps=10,
        )
    else:
        actions = _generate_action_sequence(mj_model, commands, control_dt=control_dt)
        print(f"Generated sinusoidal action sequence: T={actions.shape[0]}, nu={actions.shape[1]}")
        print("Rolling out reference trajectory...")
        ref = collect_reference_trajectory(
            mj_model=mj_model,
            param_space=ps,
            p_normalized=p_norm,
            actions=actions,
            n_substeps=10,
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    ref.save(str(out))
    print(f"Reference trajectory saved to {out}")
    print(f"  T={ref.q.shape[0]}, nq={ref.q.shape[1]}, nu={ref.actions.shape[1]}")


def cmd_identify(args):
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.sysid import SysIdConfig, ReferenceTrajectory, run_identification

    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(args.param_space)
          if args.param_space else build_t1_param_space(mj_model))

    if args.friction_only:
        ps = ps.select_by_group(["friction"])
        print(f"Using friction-only param space (d={ps.d})")

    print(f"Parameter space: d={ps.d}")

    cfg = SysIdConfig.from_yaml(args.config) if args.config else SysIdConfig()
    ref = ReferenceTrajectory.load(args.reference)
    print(f"Reference trajectory: T={ref.q.shape[0]} steps")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting CMA-ES (popsize={cfg.popsize}, gen={cfg.num_generations})...")
    p_star, Sigma, info = run_identification(
        mj_model=mj_model,
        param_space=ps,
        ref=ref,
        config=cfg,
    )

    np.save(str(out_dir / "p_star.npy"), np.array(p_star))
    np.save(str(out_dir / "Sigma.npy"), np.array(Sigma))
    ps.save(str(out_dir / "param_space.json"))

    summary = {
        "final_loss": float(info.get("final_loss", float("nan"))),
        "num_generations_run": int(info.get("num_generations_run", 0)),
        "sigma_trace": float(jnp.trace(Sigma)),
        "sigma_condition_number": float(
            jnp.max(jnp.linalg.eigvalsh(Sigma)) /
            jnp.clip(jnp.min(jnp.linalg.eigvalsh(Sigma)), 1e-10, None)
        ),
    }
    (out_dir / "identification_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "sysid_info.json").write_text(json.dumps(info, indent=2, default=float))

    print(f"\nIdentification complete.")
    print(f"  Final loss:  {summary['final_loss']:.6f}")
    print(f"  Generations: {summary['num_generations_run']}")
    print(f"  Σ trace:     {summary['sigma_trace']:.4f}")
    print(f"  Results →    {out_dir}")


def cmd_verify(args):
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.evaluate import compute_covariance_calibration, compute_param_recovery

    rd = Path(args.results_dir)
    if not (rd / "p_true.npy").exists():
        print("ERROR: p_true.npy not found. Run create-sim-a first.")
        sys.exit(1)

    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(str(rd / "param_space.json"))
          if (rd / "param_space.json").exists()
          else build_t1_param_space(mj_model))

    p_star = jnp.array(np.load(rd / "p_star.npy"))
    Sigma = jnp.array(np.load(rd / "Sigma.npy"))
    p_true = jnp.array(np.load(rd / "p_true.npy"))

    if p_star.shape[0] != p_true.shape[0]:
        raise ValueError(
            f"Shape mismatch: p_star has d={p_star.shape[0]} but p_true has d={p_true.shape[0]}. "
            "Make sure the run directory contains the matching param_space.json and p_true.npy."
        )

    recovery = compute_param_recovery(p_star, p_true, ps)
    calibration = compute_covariance_calibration(p_star, p_true, Sigma, ps)

    print("=" * 55)
    print("Parameter Recovery:")
    print(f"  Total RMSE:   {recovery['total_rmse']:.4f}")
    for g, v in recovery.get("per_group", {}).items():
        print(f"  {g:12s}:   {v['rmse']:.4f}")
    print()
    print("Covariance Calibration:")
    print(f"  Pearson r:    {calibration['pearson_correlation']:.3f}")
    print(f"  Spearman r:   {calibration['spearman_correlation']:.3f}")
    print(f"  → {calibration['interpretation']}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**calibration, "recovery": recovery}, indent=2))
        print(f"\nSaved to {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PGDR System Identification pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command")

    p = subs.add_parser("sensitivity", help="One-at-a-time sensitivity analysis")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--perturb", type=float, default=0.2)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--output", default="pgdr/results/sensitivity.json")

    p = subs.add_parser("create-sim-a", help="Create Sim A ground-truth parameters")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--perturb", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--param-space", default=None)
    p.add_argument("--friction-only", action="store_true",
                   help="Only perturb friction (dof_damping) parameters")
    p.add_argument("--output-dir", default="pgdr/results/")

    p = subs.add_parser("collect-reference", help="Collect reference trajectories")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--sim-a-params", default=None,
                   help="p_true.npy from create-sim-a (None = default model)")
    p.add_argument("--param-space", default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--onnx-policy", default=None, metavar="PATH",
                   help="Path to ONNX policy file.  If provided, uses closed-loop "
                        "ONNX rollout instead of sinusoidal actions.")
    p.add_argument("--output", default="pgdr/results/reference.npz")

    p = subs.add_parser("identify", help="Run CMA-ES identification → p*, Σ")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--param-space", default=None)
    p.add_argument("--friction-only", action="store_true",
                   help="Identify only friction (dof_damping) parameters. "
                        "Ignored if --param-space already points to a friction-only space.")
    p.add_argument("--config", default="pgdr/config/sysid_config.yaml")
    p.add_argument("--output-dir", default="pgdr/results/")

    p = subs.add_parser("verify", help="Verify covariance calibration (sim-to-sim)")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--results-dir", default="pgdr/results/")
    p.add_argument("--output", default=None)

    args = parser.parse_args()
    dispatch = {
        "sensitivity": cmd_sensitivity,
        "create-sim-a": cmd_create_sim_a,
        "collect-reference": cmd_collect_reference,
        "identify": cmd_identify,
        "verify": cmd_verify,
    }
    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

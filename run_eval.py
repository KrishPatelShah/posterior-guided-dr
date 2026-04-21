"""
run_eval.py — Entry point for PGDR evaluation.

Evaluates trained policies across conditions and generates all
analysis plots.

Typical workflow:

  # Verify covariance calibration (sim-to-sim, requires p_true)
  python run_eval.py calibration \\
      --model-xml path/to/t1.xml \\
      --results-dir pgdr/results/ \\
      --output pgdr/results/calibration.json

  # Evaluate all trained policies (velocity tracking)
  python run_eval.py sim2sim \\
      --model-xml path/to/t1.xml \\
      --results-dir pgdr/results/ \\
      --checkpoints pgdr/checkpoints/ \\
      --output pgdr/results/eval_results.json

  # Generate all plots from results
  python run_eval.py plot \\
      --results-dir pgdr/results/ \\
      --output-dir pgdr/figures/
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

import jax.numpy as jnp
import numpy as np

from pgdr.model_utils import load_mj_model, resolve_param_space_path


def cmd_calibration(args):
    """Check whether CMA-ES covariance reflects true identification uncertainty."""
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.evaluate import compute_covariance_calibration, compute_param_recovery

    if not Path(args.results_dir, "p_true.npy").exists():
        print("ERROR: p_true.npy not found in --results-dir.")
        print("This check requires sim-to-sim mode (run run_sysid.py create-sim-a first).")
        sys.exit(1)

    param_space_path = resolve_param_space_path(args.param_space, args.results_dir)
    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(param_space_path)
          if param_space_path else build_t1_param_space(mj_model))

    rd = Path(args.results_dir)
    p_star = jnp.array(np.load(rd / "p_star.npy"))
    Sigma = jnp.array(np.load(rd / "Sigma.npy"))
    p_true = jnp.array(np.load(rd / "p_true.npy"))

    if p_star.shape[0] != p_true.shape[0]:
        raise ValueError(
            f"Shape mismatch: p_star has d={p_star.shape[0]} but p_true has d={p_true.shape[0]}. "
            "Pass the matching reduced param_space.json or regenerate a clean run directory."
        )

    recovery = compute_param_recovery(p_star, p_true, ps)
    calibration = compute_covariance_calibration(p_star, p_true, Sigma, ps)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({**calibration, "recovery": recovery}, indent=2))

    print("=" * 55)
    print("Covariance Calibration Results")
    print("=" * 55)
    print(f"Parameter recovery RMSE:  {recovery['total_rmse']:.4f}")
    print(f"Pearson correlation:      {calibration['pearson_correlation']:.3f}")
    print(f"Spearman correlation:     {calibration['spearman_correlation']:.3f}")
    print(f"Interpretation:           {calibration['interpretation']}")
    print()
    print("Per-group recovery RMSE:")
    for g, v in recovery.get("per_group", {}).items():
        print(f"  {g:12s}: {v['rmse']:.4f}")
    print()
    print("Per-group calibration correlation:")
    for g, v in calibration.get("per_group_correlation", {}).items():
        print(f"  {g:12s}: {v:.3f}")
    print(f"\nResults saved to {out_path}")


def cmd_sim2sim(args):
    """Evaluate all trained policies under nominal and perturbed conditions."""
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.evaluate import evaluate_all_conditions

    param_space_path = resolve_param_space_path(args.param_space, args.results_dir)
    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(param_space_path)
          if param_space_path else build_t1_param_space(mj_model))

    rd = Path(args.results_dir)
    p_star = jnp.array(np.load(rd / "p_star.npy"))
    Sigma = jnp.array(np.load(rd / "Sigma.npy"))
    p_true = (jnp.array(np.load(rd / "p_true.npy"))
              if (rd / "p_true.npy").exists() else None)

    command_sequence = [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
    ]

    print(f"Evaluating policies in {args.checkpoints} ...")
    results = evaluate_all_conditions(
        mj_model=mj_model,
        param_space=ps,
        p_star=p_star,
        Sigma=Sigma,
        p_true=p_true,
        checkpoints_dir=args.checkpoints,
        command_sequence=command_sequence,
        num_episodes=args.num_episodes,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {out_path}")


def cmd_param_sweep(args):
    """Sweep N(p*, αΣ) uncertainty scale — the core PGDR robustness eval."""
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.evaluate import run_param_perturbation_sweep

    param_space_path = resolve_param_space_path(args.param_space, args.results_dir)
    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(param_space_path)
          if param_space_path else build_t1_param_space(mj_model))

    rd = Path(args.results_dir)
    p_star = jnp.array(np.load(rd / "p_star.npy"))
    Sigma = jnp.array(np.load(rd / "Sigma.npy"))

    alpha_levels = [float(x) for x in args.alpha_levels.split(",")]
    conditions = args.conditions.split(",") if args.conditions else None

    command_sequence = [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
    ]

    results = run_param_perturbation_sweep(
        mj_model=mj_model,
        param_space=ps,
        p_star=p_star,
        Sigma=Sigma,
        checkpoints_dir=args.checkpoints,
        command_sequence=command_sequence,
        alpha_levels=alpha_levels,
        conditions_to_eval=conditions,
        n_param_samples=args.n_param_samples,
        num_episodes=args.num_episodes,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved parameter sweep to {out_path}")


def cmd_robustness(args):
    """Sweep payload perturbation to compare C2 vs C4 robustness."""
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.evaluate import run_payload_sweep

    param_space_path = resolve_param_space_path(args.param_space, args.results_dir)
    mj_model = load_mj_model(args.model_xml)
    ps = (ParamSpace.load(param_space_path)
          if param_space_path else build_t1_param_space(mj_model))

    rd = Path(args.results_dir)
    p_star = jnp.array(np.load(rd / "p_star.npy"))

    payload_levels = [float(x) for x in args.payload_levels.split(",")]
    conditions = args.conditions.split(",") if args.conditions else None

    command_sequence = [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
    ]

    results = run_payload_sweep(
        mj_model=mj_model,
        param_space=ps,
        p_star=p_star,
        checkpoints_dir=args.checkpoints,
        command_sequence=command_sequence,
        payload_levels=payload_levels,
        conditions_to_eval=conditions,
        num_episodes=args.num_episodes,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved robustness sweep to {out_path}")


def cmd_plot(args):
    """Generate all analysis plots."""
    from pgdr.plotting import generate_all_plots
    import shutil

    rd = Path(args.results_dir)
    od = Path(args.output_dir)
    od.mkdir(parents=True, exist_ok=True)

    if not any(rd.glob("*.json")) and not any(rd.glob("*.npy")):
        print(f"No result files found in {rd}")
        sys.exit(1)

    print(f"Generating plots from {rd} → {od} ...")
    # plotting.py reads from results_dir and writes figures alongside results;
    # copy to output_dir if different
    generate_all_plots(str(rd))

    # Move generated figures to output_dir if distinct
    if rd.resolve() != od.resolve():
        for fig in rd.glob("*.png"):
            shutil.move(str(fig), str(od / fig.name))
        for fig in rd.glob("*.pdf"):
            shutil.move(str(fig), str(od / fig.name))

    print(f"Plots saved to {od}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PGDR Evaluation & Plotting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command")

    # -- calibration --
    p = subs.add_parser("calibration",
                        help="Check covariance calibration (sim-to-sim)")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--results-dir", default="pgdr/results/")
    p.add_argument("--param-space", default=None)
    p.add_argument("--output", default="pgdr/results/calibration.json")

    # -- sim2sim --
    p = subs.add_parser("sim2sim",
                        help="Evaluate trained policies (velocity tracking)")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--results-dir", default="pgdr/results/")
    p.add_argument("--checkpoints", default="pgdr/checkpoints/")
    p.add_argument("--param-space", default=None)
    p.add_argument("--num-episodes", type=int, default=50)
    p.add_argument("--output", default="pgdr/results/eval_results.json")

    # -- param-sweep --
    p = subs.add_parser("param-sweep",
                        help="Sweep N(p*, αΣ) uncertainty scale (core PGDR eval)")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--results-dir", default="pgdr/results/")
    p.add_argument("--checkpoints", default="pgdr/checkpoints/")
    p.add_argument("--param-space", default=None)
    p.add_argument("--alpha-levels", default="0,0.25,0.5,1.0,1.5,2.0,3.0",
                   help="Comma-separated α values (scales Σ)")
    p.add_argument("--conditions", default="C1_uniform_dr,C2_pure_sysid,C3_isotropic,C4_pgdr_1.0",
                   help="Comma-separated condition names to evaluate")
    p.add_argument("--n-param-samples", type=int, default=20,
                   help="Parameter vectors sampled per α level")
    p.add_argument("--num-episodes", type=int, default=5)
    p.add_argument("--output", default="pgdr/results/param_sweep.json")

    # -- robustness --
    p = subs.add_parser("robustness",
                        help="Sweep payload perturbation to compare condition robustness")
    p.add_argument("--model-xml", required=True)
    p.add_argument("--results-dir", default="pgdr/results/")
    p.add_argument("--checkpoints", default="pgdr/checkpoints/")
    p.add_argument("--param-space", default=None)
    p.add_argument("--payload-levels", default="0,0.5,1.0,1.5,2.0,2.5,3.0",
                   help="Comma-separated payload values in kg")
    p.add_argument("--conditions", default="C2_pure_sysid,C4_pgdr_1.0",
                   help="Comma-separated condition names to evaluate")
    p.add_argument("--num-episodes", type=int, default=10)
    p.add_argument("--output", default="pgdr/results/robustness_sweep.json")

    # -- plot --
    p = subs.add_parser("plot", help="Generate all analysis plots")
    p.add_argument("--results-dir", default="pgdr/results/")
    p.add_argument("--output-dir", default="pgdr/figures/")

    args = parser.parse_args()

    dispatch = {
        "calibration": cmd_calibration,
        "sim2sim": cmd_sim2sim,
        "param-sweep": cmd_param_sweep,
        "robustness": cmd_robustness,
        "plot": cmd_plot,
    }

    if args.command not in dispatch:
        parser.print_help()
        sys.exit(1)

    dispatch[args.command](args)


if __name__ == "__main__":
    main()

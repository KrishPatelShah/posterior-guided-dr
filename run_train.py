"""
run_train.py — Entry point for PGDR policy training.

Trains locomotion policies under all four experimental conditions
(C1–C4) using PPO with the PGDREnv wrapper.

Typical workflow:

  # Train all conditions on all seeds (sequential)
  python run_train.py \\
      --model-xml path/to/t1.xml \\
      --results-dir pgdr/results/ \\
      --checkpoint-dir pgdr/checkpoints/

  # Train a single condition for debugging
  python run_train.py \\
      --model-xml path/to/t1.xml \\
      --results-dir pgdr/results/ \\
      --checkpoint-dir pgdr/checkpoints/ \\
      --conditions C4_pgdr_1.0 \\
      --seeds 0

  # Dry run (print plan without training)
  python run_train.py --model-xml path/to/t1.xml \\
      --results-dir pgdr/results/ --dry-run

  # Generate SLURM scripts for TACC
  python run_train.py slurm \\
      --model-xml path/to/t1.xml \\
      --results-dir pgdr/results/ \\
      --checkpoint-dir pgdr/checkpoints/ \\
      --slurm-dir pgdr/tacc/jobs/
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="PGDR Policy Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = parser.add_subparsers(dest="command")

    # -- train (default) --
    p_train = subs.add_parser("train", help="Run training for all conditions")
    p_train.add_argument("--model-xml", required=True, help="Path to T1 XML model")
    p_train.add_argument("--config", default="pgdr/config/train_config.yaml")
    p_train.add_argument("--results-dir", default="pgdr/results/",
                         help="Directory with p_star.npy and Sigma.npy from run_sysid.py")
    p_train.add_argument("--param-space", default=None,
                         help="Path to param_space.json (uses full space if omitted)")
    p_train.add_argument("--checkpoint-dir", default="pgdr/checkpoints/")
    p_train.add_argument("--conditions", nargs="+", default=None,
                         help="Subset of conditions to train (e.g. C4_pgdr_1.0 C1_uniform_dr)")
    p_train.add_argument("--seeds", nargs="+", type=int, default=None,
                         help="Seeds to train (default: from config, usually 0 1 2)")
    p_train.add_argument("--dry-run", action="store_true",
                         help="Print training plan without running")

    # -- slurm --
    p_slurm = subs.add_parser("slurm", help="Generate SLURM scripts for TACC Lonestar6")
    p_slurm.add_argument("--model-xml", required=True)
    p_slurm.add_argument("--config", default="pgdr/config/train_config.yaml")
    p_slurm.add_argument("--results-dir", default="pgdr/results/")
    p_slurm.add_argument("--checkpoint-dir", default="pgdr/checkpoints/")
    p_slurm.add_argument("--slurm-dir", default="pgdr/tacc/jobs/")

    args = parser.parse_args()

    # Default to "train" subcommand if not specified
    if args.command is None:
        # Re-parse with train as default
        args = parser.parse_args(["train"] + sys.argv[1:])

    if args.command == "train":
        from pgdr.train_all_conditions import train_all
        train_all(args)

    elif args.command == "slurm":
        from pgdr.train_all_conditions import generate_slurm_jobs
        generate_slurm_jobs(args)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

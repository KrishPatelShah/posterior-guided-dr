#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np

if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True)


def run_warp_smoke(model_xml: str, output_dir: Path) -> Path:
    from pgdr.model_utils import load_mj_model
    from pgdr.param_space import build_t1_param_space
    from pgdr.sysid import collect_reference_trajectory

    mj_model = load_mj_model(model_xml)
    ps = build_t1_param_space(mj_model)
    actions = np.zeros((50, mj_model.nu), dtype=np.float32)
    ref = collect_reference_trajectory(
        mj_model=mj_model,
        param_space=ps,
        p_normalized=np.zeros(ps.d, dtype=np.float32),
        actions=actions,
        n_substeps=2,
    )
    out = output_dir / "warp_smoke_reference.npz"
    ref.save(str(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PGDR pipeline end to end.")
    parser.add_argument("--model-xml", default="playground:t1")
    parser.add_argument("--sysid-config", default="pgdr/config/sysid_config_smoke.yaml")
    parser.add_argument("--train-config", default="pgdr/config/train_config_smoke.yaml")
    parser.add_argument("--results-root", default="pgdr/results")
    parser.add_argument("--checkpoints-root", default="pgdr/checkpoints")
    parser.add_argument("--num-eval-episodes", type=int, default=5)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (repo_root / args.results_root / f"{timestamp}_auto").resolve()
    ckpt_dir = (repo_root / args.checkpoints_root / run_dir.name).resolve()
    fig_dir = run_dir / "figures"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    commands = [
        [
            "uv", "run", "python", "run_sysid.py", "create-sim-a",
            "--model-xml", args.model_xml,
            "--output-dir", str(run_dir),
        ],
        [
            "uv", "run", "python", "run_sysid.py", "collect-reference",
            "--model-xml", args.model_xml,
            "--sim-a-params", str(run_dir / "p_true.npy"),
            "--param-space", str(run_dir / "param_space.json"),
            "--config", args.sysid_config,
            "--output", str(run_dir / "reference.npz"),
        ],
        [
            "uv", "run", "python", "run_sysid.py", "identify",
            "--model-xml", args.model_xml,
            "--reference", str(run_dir / "reference.npz"),
            "--param-space", str(run_dir / "param_space.json"),
            "--config", args.sysid_config,
            "--output-dir", str(run_dir),
        ],
        [
            "uv", "run", "python", "run_eval.py", "calibration",
            "--model-xml", args.model_xml,
            "--results-dir", str(run_dir),
            "--param-space", str(run_dir / "param_space.json"),
            "--output", str(run_dir / "calibration.json"),
        ],
        [
            "uv", "run", "python", "run_train.py", "train",
            "--model-xml", args.model_xml,
            "--config", args.train_config,
            "--results-dir", str(run_dir),
            "--param-space", str(run_dir / "param_space.json"),
            "--checkpoint-dir", str(ckpt_dir),
        ],
        [
            "uv", "run", "python", "run_eval.py", "sim2sim",
            "--model-xml", args.model_xml,
            "--results-dir", str(run_dir),
            "--param-space", str(run_dir / "param_space.json"),
            "--checkpoints", str(ckpt_dir),
            "--num-episodes", str(args.num_eval_episodes),
            "--output", str(run_dir / "eval_results.json"),
        ],
        [
            "uv", "run", "python", "run_eval.py", "plot",
            "--results-dir", str(run_dir),
            "--output-dir", str(fig_dir),
        ],
    ]

    for cmd in commands:
        run_cmd(cmd, repo_root)

    warp_smoke_path = run_warp_smoke(args.model_xml, run_dir)

    summary = {
        "run_dir": str(run_dir),
        "checkpoints_dir": str(ckpt_dir),
        "figures_dir": str(fig_dir),
        "warp_smoke_reference": str(warp_smoke_path),
    }
    (run_dir / "pipeline_summary.json").write_text(json.dumps(summary, indent=2))

    print("\nPipeline finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

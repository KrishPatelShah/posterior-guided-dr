#!/usr/bin/env python3
"""
run_fim_analysis.py — Fisher Information Matrix analysis for T1 parameter identification.

Determines which parameters are identifiable from ONNX policy trajectories by
computing the FIM diagonal for a set of command sequences.  Optionally runs
CMA-ES to find the command sequence that maximizes FIM for a target parameter
group.

The FIM diagonal for parameter i under a given action sequence is:
    FIM_i = max(||Δq_i||², ||Δq_i⁻||²) / (perturbation² · T)

where Δq_i is the trajectory difference when parameter i is perturbed by
±perturbation in normalized space.  High FIM → identifiable.  Low FIM →
not identifiable regardless of how long CMA-ES runs.

Usage:
    # Score standard command sequences for friction params
    python run_fim_analysis.py \\
        --model-xml t1 \\
        --onnx-policy .venv/lib/python3.11/site-packages/mujoco_playground/experimental/sim2sim/onnx/t1_policy.onnx \\
        --groups friction

    # Optimize command sequence to maximize friction FIM
    python run_fim_analysis.py \\
        --model-xml t1 \\
        --onnx-policy <path> \\
        --groups friction \\
        --optimize \\
        --output-dir pgdr/results/fim_friction
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

if platform.system() == "Darwin":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from pgdr.model_utils import load_mj_model
from pgdr.param_space import build_t1_param_space
from pgdr.sensitivity import run_sensitivity_analysis
from pgdr.sysid import (
    collect_reference_from_onnx_policy,
    load_onnx_policy,
)


# ---------------------------------------------------------------------------
# Standard command sequences to evaluate
# ---------------------------------------------------------------------------

CANDIDATE_COMMANDS = {
    "stand":         [{"vx": 0.0,  "vy": 0.0, "wz": 0.0,  "duration": 8.0}],
    "walk_slow":     [{"vx": 0.5,  "vy": 0.0, "wz": 0.0,  "duration": 8.0}],
    "walk_fast":     [{"vx": 1.5,  "vy": 0.0, "wz": 0.0,  "duration": 8.0}],
    "walk_back":     [{"vx": -0.5, "vy": 0.0, "wz": 0.0,  "duration": 8.0}],
    "turn":          [{"vx": 0.0,  "vy": 0.0, "wz": 1.0,  "duration": 8.0}],
    "walk_and_turn": [
        {"vx": 1.0,  "vy": 0.0, "wz": 0.0,  "duration": 4.0},
        {"vx": 0.0,  "vy": 0.0, "wz": 1.0,  "duration": 4.0},
    ],
    "mixed": [
        {"vx": 1.0,  "vy": 0.0, "wz": 0.0,  "duration": 3.0},
        {"vx": 0.0,  "vy": 0.0, "wz": 1.0,  "duration": 2.0},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0,  "duration": 2.0},
        {"vx": 0.0,  "vy": 0.0, "wz": 0.0,  "duration": 1.0},
    ],
}

# Command bounds for optimization (physical units)
CMD_BOUNDS = {
    "vx":  (-1.5, 1.5),
    "vy":  (-0.5, 0.5),
    "wz":  (-1.0, 1.0),
}


# ---------------------------------------------------------------------------
# Core FIM computation
# ---------------------------------------------------------------------------

def compute_fim_for_commands(
    mj_model,
    param_space,
    commands: list[dict],
    onnx_session,
    onnx_input_name: str,
    onnx_output_name: str,
    perturbation: float = 0.2,
    control_dt: float = 0.02,
    n_substeps: int = 10,
    w_q: float = 1.0,
    w_qdot: float = 1.0,
) -> dict[str, float]:
    """
    Generate an ONNX policy trajectory for the given commands, then compute
    the FIM diagonal via one-at-a-time finite-difference perturbations.

    Returns a dict mapping param name → FIM score (higher = more identifiable).
    The FIM score is the normalised trajectory divergence:
        FIM_i = sensitivity_score_i / perturbation²
    """
    # Generate reference trajectory with default (unperturbed) model
    ref = collect_reference_from_onnx_policy(
        mj_model, param_space,
        np.zeros(param_space.d),          # nominal params
        onnx_session, onnx_input_name, onnx_output_name,
        commands, control_dt, n_substeps,
    )

    results = run_sensitivity_analysis(
        mj_model, param_space, ref.actions,
        perturbation=perturbation,
        n_substeps=n_substeps,
        w_q=w_q,
        w_qdot=w_qdot,
    )

    # Normalise by perturbation² so scores are comparable across delta values
    fim = {
        name: score / (perturbation ** 2)
        for name, score in results["sensitivity_scores"].items()
    }
    return fim


# ---------------------------------------------------------------------------
# Command optimization (CMA-ES over K-segment command space)
# ---------------------------------------------------------------------------

def _encode_commands(x: np.ndarray, n_segments: int, duration_per_seg: float) -> list[dict]:
    """Decode a flat normalized vector x ∈ [-1,1]^(3K) into a command list."""
    commands = []
    for k in range(n_segments):
        vx = float(x[3*k]   * CMD_BOUNDS["vx"][1])
        vy = float(x[3*k+1] * CMD_BOUNDS["vy"][1])
        wz = float(x[3*k+2] * CMD_BOUNDS["wz"][1])
        commands.append({"vx": vx, "vy": vy, "wz": wz, "duration": duration_per_seg})
    return commands


def optimize_commands(
    mj_model,
    param_space,
    onnx_session,
    onnx_input_name: str,
    onnx_output_name: str,
    target_groups: list[str],
    n_segments: int = 4,
    total_duration: float = 8.0,
    perturbation: float = 0.2,
    control_dt: float = 0.02,
    n_substeps: int = 10,
    w_q: float = 1.0,
    w_qdot: float = 1.0,
    popsize: int = 8,
    max_generations: int = 30,
    seed: int = 42,
) -> tuple[list[dict], dict[str, float], dict]:
    """
    CMA-ES over a K-segment piecewise-constant command space to maximise
    the sum of FIM scores for the target parameter groups.

    Returns (best_commands, best_fim_scores, history).
    """
    from cmaes import CMA

    duration_per_seg = total_duration / n_segments
    d_cmd = 3 * n_segments                    # optimization dimension
    target_params = [p.name for p in param_space.params if p.group in target_groups]

    bounds = np.array([[-1.0, 1.0]] * d_cmd)
    optimizer = CMA(
        mean=np.zeros(d_cmd),
        sigma=0.5,
        population_size=popsize,
        seed=seed,
        bounds=bounds,
    )

    history = {"generation": [], "best_fim_trace": [], "mean_fim_trace": []}
    best_trace = -np.inf
    best_commands = None
    best_fim = None

    print(f"\nOptimizing {d_cmd}-dim command space "
          f"({n_segments} segments × 3 commands) "
          f"for {target_groups} FIM...")
    print(f"  popsize={popsize}, max_gen={max_generations}\n")

    for gen in range(max_generations):
        solutions = [optimizer.ask() for _ in range(popsize)]
        traces = []

        for x in solutions:
            commands = _encode_commands(x, n_segments, duration_per_seg)
            try:
                fim = compute_fim_for_commands(
                    mj_model, param_space, commands,
                    onnx_session, onnx_input_name, onnx_output_name,
                    perturbation=perturbation,
                    control_dt=control_dt,
                    n_substeps=n_substeps,
                    w_q=w_q, w_qdot=w_qdot,
                )
                trace = float(np.sum([fim[p] for p in target_params]))
            except Exception:
                trace = 0.0
            traces.append(trace)

        # CMA-ES minimises — negate for maximisation
        optimizer.tell([(solutions[i], -traces[i]) for i in range(popsize)])

        gen_best = float(np.max(traces))
        gen_mean = float(np.mean(traces))
        history["generation"].append(gen)
        history["best_fim_trace"].append(gen_best)
        history["mean_fim_trace"].append(gen_mean)

        if gen_best > best_trace:
            best_trace = gen_best
            best_commands = _encode_commands(solutions[int(np.argmax(traces))], n_segments, duration_per_seg)
            best_fim = compute_fim_for_commands(
                mj_model, param_space, best_commands,
                onnx_session, onnx_input_name, onnx_output_name,
                perturbation=perturbation, control_dt=control_dt,
                n_substeps=n_substeps, w_q=w_q, w_qdot=w_qdot,
            )

        print(f"  Gen {gen:3d}: best_trace={gen_best:.4f}  mean={gen_mean:.4f}")

    return best_commands, best_fim, history


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_fim_table(
    fim_results: dict[str, dict[str, float]],
    param_space,
    threshold_frac: float = 0.1,
) -> None:
    """
    Print a table of FIM scores per parameter per command sequence.
    Marks parameters as identifiable (above threshold_frac of max FIM seen).
    """
    all_scores = [s for scores in fim_results.values() for s in scores.values()]
    max_fim = max(all_scores) if all_scores else 1.0
    threshold = threshold_frac * max_fim

    cmd_names = list(fim_results.keys())
    param_names = [p.name for p in param_space.params]

    col_w = 12
    header = f"{'Parameter':<35} {'Group':<12}" + "".join(f"{n:>{col_w}}" for n in cmd_names)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for param in param_space.params:
        scores = [fim_results[cmd].get(param.name, 0.0) for cmd in cmd_names]
        best = max(scores)
        flag = "  ✓" if best >= threshold else "  ✗"
        row = f"{param.name:<35} {param.group:<12}"
        for s in scores:
            bar = "█" if s >= threshold else "·"
            row += f"{s:>{col_w-1}.4f}{bar}"
        print(row + flag)

    print("=" * len(header))
    print(f"\n  ✓ = identifiable (FIM ≥ {threshold:.4f} = {threshold_frac*100:.0f}% of max)")
    print(f"  ✗ = not identifiable from any tested command sequence\n")


def summarise_by_group(
    fim_results: dict[str, dict[str, float]],
    param_space,
) -> dict[str, dict]:
    """Return per-group best FIM and best command for each group."""
    groups = sorted(set(p.group for p in param_space.params))
    summary = {}
    for g in groups:
        group_params = [p.name for p in param_space.params if p.group == g]
        best_cmd, best_trace = None, -np.inf
        for cmd_name, fim in fim_results.items():
            trace = sum(fim.get(p, 0.0) for p in group_params)
            if trace > best_trace:
                best_trace = trace
                best_cmd = cmd_name
        avg_fim = np.mean([
            max(fim_results[cmd].get(p, 0.0) for cmd in fim_results)
            for p in group_params
        ])
        summary[g] = {"best_command": best_cmd, "best_trace": best_trace, "avg_peak_fim": avg_fim}
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIM analysis — which T1 parameters are identifiable from ONNX trajectories?",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model-xml", default="t1")
    parser.add_argument("--onnx-policy", required=True, metavar="PATH")
    parser.add_argument(
        "--groups", nargs="+", default=["friction"],
        help="Parameter groups to analyse (default: friction)",
    )
    parser.add_argument("--perturbation", type=float, default=0.2,
                        help="Finite-difference delta in normalised param space (default: 0.2)")
    parser.add_argument("--w-qdot", type=float, default=1.0,
                        help="Weight on velocity divergence (default: 1.0 — friction-sensitive)")
    parser.add_argument(
        "--optimize", action="store_true",
        help="Run CMA-ES to find optimal command sequence for target groups",
    )
    parser.add_argument("--opt-segments", type=int, default=4,
                        help="Number of command segments for optimization (default: 4)")
    parser.add_argument("--opt-generations", type=int, default=30,
                        help="CMA-ES generations for command optimization (default: 30)")
    parser.add_argument("--opt-popsize", type=int, default=8)
    parser.add_argument(
        "--output-dir", default=None,
        help="Save results JSON and plots here (default: pgdr/results/fim_<timestamp>)",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"pgdr/results/fim_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"FIM Analysis — parameter identifiability")
    print(f"{'='*60}")
    print(f"  Model:       {args.model_xml}")
    print(f"  Policy:      {args.onnx_policy}")
    print(f"  Groups:      {args.groups}")
    print(f"  Perturbation:{args.perturbation}  w_qdot:{args.w_qdot}")
    print()

    mj_model = load_mj_model(args.model_xml)
    ps_full = build_t1_param_space(mj_model)

    if "all" not in args.groups:
        ps = ps_full.select_by_group(args.groups)
    else:
        ps = ps_full

    print(f"  Parameter space: d={ps.d}  "
          f"{dict((g, len(ps.group_indices(g))) for g in set(p.group for p in ps.params))}\n")

    session, input_name, output_name = load_onnx_policy(args.onnx_policy)

    # ------------------------------------------------------------------ #
    # Evaluate standard command sequences
    # ------------------------------------------------------------------ #
    print("Evaluating standard command sequences...\n")
    fim_results: dict[str, dict[str, float]] = {}

    for cmd_name, commands in CANDIDATE_COMMANDS.items():
        total_dur = sum(c["duration"] for c in commands)
        print(f"  [{cmd_name}]  ({total_dur:.0f}s)  ", end="", flush=True)
        fim = compute_fim_for_commands(
            mj_model, ps, commands,
            session, input_name, output_name,
            perturbation=args.perturbation,
            w_q=1.0, w_qdot=args.w_qdot,
        )
        fim_results[cmd_name] = fim
        trace = sum(fim.values())
        print(f"FIM trace = {trace:.4f}")

    print_fim_table(fim_results, ps)

    group_summary = summarise_by_group(fim_results, ps)
    print("Per-group summary:")
    for g, info in group_summary.items():
        print(f"  {g:<12}  best_cmd={info['best_command']:<16}  "
              f"trace={info['best_trace']:.4f}  avg_peak_FIM={info['avg_peak_fim']:.4f}")

    # ------------------------------------------------------------------ #
    # Optional: CMA-ES command optimization
    # ------------------------------------------------------------------ #
    opt_results = None
    if args.optimize:
        best_cmds, best_fim, opt_history = optimize_commands(
            mj_model, ps,
            session, input_name, output_name,
            target_groups=args.groups,
            n_segments=args.opt_segments,
            perturbation=args.perturbation,
            w_q=1.0, w_qdot=args.w_qdot,
            popsize=args.opt_popsize,
            max_generations=args.opt_generations,
        )
        fim_results["optimized"] = best_fim
        opt_results = {"best_commands": best_cmds, "history": opt_history}

        print("\nOptimal command sequence:")
        for i, cmd in enumerate(best_cmds):
            print(f"  Segment {i+1}: vx={cmd['vx']:+.2f}  vy={cmd['vy']:+.2f}  "
                  f"wz={cmd['wz']:+.2f}  duration={cmd['duration']:.1f}s")

        print("\nFIM after optimization:")
        print_fim_table({"optimized": best_fim}, ps)

    # ------------------------------------------------------------------ #
    # Save results
    # ------------------------------------------------------------------ #
    output = {
        "fim_by_command": fim_results,
        "group_summary": group_summary,
        "config": {
            "groups": args.groups,
            "perturbation": args.perturbation,
            "w_qdot": args.w_qdot,
        },
    }
    if opt_results:
        output["optimization"] = opt_results

    (out_dir / "fim_results.json").write_text(json.dumps(output, indent=2, default=float))
    print(f"\nResults saved to {out_dir}/fim_results.json")


if __name__ == "__main__":
    main()

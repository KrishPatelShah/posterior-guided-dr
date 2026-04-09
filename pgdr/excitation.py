"""
pgdr/excitation.py — FIM-optimal sinusoidal excitation design.

Replaces the cheetah branch's fixed sinusoidal excitation with an
optimized signal that maximizes the Fisher Information Matrix (FIM)
trace for target parameter groups.

Excitation model (shared frequency):
    ctrl[t, j] = amplitude[j] * sin(2π * freq * t + phase[j])

Excitation model (per-group frequency):
    ctrl[t, j] = amplitude[j] * sin(2π * freq[g(j)] * t + phase[j])

The optimization finds (freq(s), amplitudes[nu]) that maximize the
sum of FIM diagonal entries for the target parameters. Phases are
fixed to evenly-spaced values.

Key advantages over fixed sinusoidal:
- Per-joint amplitude: joints with stronger friction signal get larger
  excitation; unresponsive joints are suppressed
- Optimized frequency: finds the frequency where friction is most
  visible in the velocity response
- Per-group frequency (optional): roll/sagittal joints can find
  independent optimal frequencies, fixing under-excitation of
  hard-to-identify joints (e.g. hip roll, ankle roll)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mujoco_warp
import warp as wp
import numpy as np

from pgdr.sensitivity import run_sensitivity_analysis


# ---------------------------------------------------------------------------
# Excitation signal generation
# ---------------------------------------------------------------------------

def generate_sinusoidal_actions(
    freq,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    duration: float,
    control_dt: float,
    nu: int,
) -> np.ndarray:
    """
    Generate a sinusoidal ctrl sequence.

    Args:
        freq:        Shared frequency (float) OR per-joint frequencies ([nu] array).
        amplitudes:  [nu] per-joint amplitude (rad).
        phases:      [nu] per-joint phase offset (rad).
        duration:    Total duration (seconds).
        control_dt:  Control timestep (seconds).
        nu:          Number of actuated joints.

    Returns:
        actions: [T, nu] ctrl array.
    """
    T = int(duration / control_dt)
    t = np.arange(T) * control_dt                              # [T]
    freqs = np.broadcast_to(np.atleast_1d(np.asarray(freq, dtype=np.float64)), (nu,))
    actions = amplitudes[None, :] * np.sin(
        2.0 * np.pi * freqs[None, :] * t[:, None] + phases[None, :]  # [T, nu]
    )
    return actions.astype(np.float64)


def cheetah_baseline_actions(
    mj_model,
    duration: float = 10.0,
    control_dt: float = 0.02,
    freq: float = 2.0,
    amplitude: float = 0.1,
) -> np.ndarray:
    """
    Reproduce the cheetah branch's fixed sinusoidal excitation:
    uniform amplitude, evenly-spaced phases, fixed frequency.
    """
    nu = mj_model.nu
    amplitudes = np.full(nu, amplitude)
    phases     = np.linspace(0, 2.0 * np.pi, nu, endpoint=False)
    return generate_sinusoidal_actions(freq, amplitudes, phases, duration, control_dt, nu)


# ---------------------------------------------------------------------------
# FIM computation for a sinusoidal excitation
# ---------------------------------------------------------------------------

def compute_sinusoidal_fim(
    mj_model,
    param_space,
    freq,  # float (shared) or [nu] array (per-joint/per-group)
    amplitudes: np.ndarray,
    phases: np.ndarray,
    duration: float = 10.0,
    control_dt: float = 0.02,
    perturbation: float = 0.2,
    n_substeps: int = 10,
    w_q: float = 1.0,
    w_qdot: float = 1.0,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Compute FIM diagonal for a sinusoidal excitation.

    Generates the action sequence, runs sensitivity analysis (batched
    warp rollout), and returns per-parameter FIM scores.

    FIM_i = sensitivity_score_i / perturbation²

    Returns:
        fim:     dict mapping param name → FIM score.
        actions: [T, nu] generated ctrl sequence.
    """
    actions = generate_sinusoidal_actions(
        freq, amplitudes, phases, duration, control_dt, mj_model.nu
    )
    results = run_sensitivity_analysis(
        mj_model, param_space, actions,
        perturbation=perturbation,
        n_substeps=n_substeps,
        w_q=w_q,
        w_qdot=w_qdot,
    )
    fim = {
        name: score / (perturbation ** 2)
        for name, score in results["sensitivity_scores"].items()
    }
    return fim, actions


# ---------------------------------------------------------------------------
# CMA-ES optimization over excitation parameters
# ---------------------------------------------------------------------------

def optimize_sinusoidal_excitation(
    mj_model,
    param_space,
    target_groups: list[str],
    duration: float = 10.0,
    control_dt: float = 0.02,
    freq_range: tuple[float, float] = (0.2, 5.0),
    amplitude_max: float = 0.3,
    perturbation: float = 0.2,
    n_substeps: int = 10,
    w_q: float = 1.0,
    w_qdot: float = 1.0,
    popsize: int = 16,
    max_generations: int = 50,
    seed: int = 42,
    freq_groups: Optional[list[list[int]]] = None,
) -> tuple[np.ndarray, dict, dict]:
    """
    CMA-ES over (freq(s), amplitudes[nu]) to maximise FIM trace for
    the target parameter groups. Phases are fixed to evenly-spaced
    values since over many periods phase offset has negligible effect.

    Optimization space (all normalized to [-1, 1]):
        Single-freq mode  (freq_groups=None):  d = 1 + nu
            x[0]         → shared frequency (log-uniform in freq_range)
            x[1:nu+1]    → amplitudes (uniform in [0, amplitude_max])

        Per-group mode    (freq_groups given):  d = n_groups + nu
            x[0:n_groups] → per-group frequencies (log-uniform)
            x[n_groups:]  → amplitudes (uniform in [0, amplitude_max])

    Args:
        freq_groups: List of lists of actuator indices, one list per
            frequency group. Joints in the same group share a frequency.
            None falls back to the original single shared frequency.

    Returns:
        best_actions: [T, nu] optimal ctrl sequence.
        best_params:  dict with freqs (per-joint array), amplitudes, phases.
                      Also contains freq_groups metadata when used.
        history:      dict with per-generation FIM traces.
    """
    from cmaes import CMA

    nu = mj_model.nu
    log_fmin = np.log(freq_range[0])
    log_fmax = np.log(freq_range[1])

    # Phases fixed to evenly-spaced — irrelevant over many periods
    fixed_phases = np.linspace(0, 2.0 * np.pi, nu, endpoint=False)

    target_params = [p.name for p in param_space.params if p.group in target_groups]

    # Build group → joint mapping
    if freq_groups is None:
        n_groups = 1
        joint_to_group = np.zeros(nu, dtype=int)
    else:
        n_groups = len(freq_groups)
        joint_to_group = np.zeros(nu, dtype=int)
        for g, indices in enumerate(freq_groups):
            for j in indices:
                joint_to_group[j] = g

    d_opt = n_groups + nu

    def decode_freq(x_g: float) -> float:
        return float(np.exp(
            log_fmin + (np.clip(x_g, -1, 1) + 1) / 2.0 * (log_fmax - log_fmin)
        ))

    def decode(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        group_freqs  = np.array([decode_freq(x[g]) for g in range(n_groups)])
        joint_freqs  = group_freqs[joint_to_group]                               # [nu]
        amplitudes   = amplitude_max * (np.clip(x[n_groups:n_groups+nu], -1, 1) + 1) / 2.0
        return joint_freqs, amplitudes, fixed_phases

    # Initialise: all groups at 2 Hz, amplitudes at amplitude_max/2
    x0_freq_val = 2.0 * (np.log(2.0) - log_fmin) / (log_fmax - log_fmin) - 1.0
    x0 = np.concatenate([
        np.full(n_groups, x0_freq_val),
        np.zeros(nu),
    ])

    bounds    = np.array([[-1.0, 1.0]] * d_opt)
    optimizer = CMA(
        mean=x0,
        sigma=0.3,
        population_size=popsize,
        seed=seed,
        bounds=bounds,
    )

    best_trace   = -np.inf
    best_actions = None
    best_params  = None
    history      = {"generation": [], "best_fim_trace": [], "mean_fim_trace": []}

    freq_desc = (f"{n_groups} group freqs" if freq_groups is not None
                 else "1 shared freq")
    print(f"\nOptimising sinusoidal excitation "
          f"(d={d_opt}: {freq_desc} + {nu} amp, phases fixed)")
    print(f"  target groups: {target_groups}")
    print(f"  popsize={popsize}, max_gen={max_generations}, "
          f"duration={duration}s, A_max={amplitude_max}rad\n")

    for gen in range(max_generations):
        solutions = [optimizer.ask() for _ in range(popsize)]
        traces    = []

        for x in solutions:
            joint_freqs, amplitudes, phases = decode(x)
            try:
                fim, _ = compute_sinusoidal_fim(
                    mj_model, param_space,
                    joint_freqs, amplitudes, phases,
                    duration, control_dt, perturbation, n_substeps,
                    w_q, w_qdot,
                )
                trace = float(np.sum([fim.get(p, 0.0) for p in target_params]))
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

        improved = gen_best > best_trace
        if improved:
            best_trace = gen_best
            best_x     = solutions[int(np.argmax(traces))]
            joint_freqs, amplitudes, phases = decode(best_x)
            best_params = {
                "freqs":      joint_freqs.tolist(),
                "amplitudes": amplitudes.tolist(),
                "phases":     phases.tolist(),
                "duration":   duration,
                "control_dt": control_dt,
            }
            if freq_groups is not None:
                best_params["freq_groups"] = freq_groups
            _, best_actions = compute_sinusoidal_fim(
                mj_model, param_space,
                joint_freqs, amplitudes, phases,
                duration, control_dt, perturbation, n_substeps,
                w_q, w_qdot,
            )

        best_x_gen    = solutions[int(np.argmax(traces))]
        best_freqs, _, _ = decode(best_x_gen)
        freq_str = (f"freqs=[{', '.join(f'{f:.2f}' for f in np.unique(best_freqs))}]Hz"
                    if freq_groups is not None
                    else f"freq={best_freqs[0]:.2f}Hz")
        marker = " *" if improved else ""
        print(f"  Gen {gen:3d}: gen_best={gen_best:.4f}  running_best={best_trace:.4f}  "
              f"mean={gen_mean:.4f}  {freq_str}{marker}")

    return best_actions, best_params, history


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_excitation(
    actions: np.ndarray,
    params: dict,
    out_dir: str,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(str(out / "excitation_actions.npy"), actions)
    (out / "excitation_params.json").write_text(json.dumps(params, indent=2))
    print(f"  Saved excitation_actions.npy + excitation_params.json to {out}/")


def load_excitation(out_dir: str) -> tuple[np.ndarray, dict]:
    out = Path(out_dir)
    actions = np.load(str(out / "excitation_actions.npy"))
    params  = json.loads((out / "excitation_params.json").read_text())
    return actions, params

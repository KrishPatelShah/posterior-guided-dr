"""
pgdr/excitation.py — FIM-optimal sinusoidal excitation design.

Replaces the cheetah branch's fixed sinusoidal excitation with an
optimized signal that maximizes the Fisher Information Matrix (FIM)
trace for target parameter groups.

Excitation model:
    ctrl[t, j] = amplitude[j] * sin(2π * freq * t + phase[j])

The optimization finds (freq, amplitudes[nu], phases[nu]) that
maximize the sum of FIM diagonal entries for the target parameters.

Key advantages over fixed sinusoidal:
- Per-joint amplitude: joints with stronger friction signal get larger
  excitation; unresponsive joints are suppressed
- Optimized frequency: finds the frequency where friction is most
  visible in the velocity response
- Per-joint phase: maximizes independence between joints so the 23
  damping values are separately identifiable
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
    freq: float,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    duration: float,
    control_dt: float,
    nu: int,
) -> np.ndarray:
    """
    Generate a sinusoidal ctrl sequence.

    Args:
        freq:        Shared oscillation frequency (Hz).
        amplitudes:  [nu] per-joint amplitude (rad).
        phases:      [nu] per-joint phase offset (rad).
        duration:    Total duration (seconds).
        control_dt:  Control timestep (seconds).
        nu:          Number of actuated joints.

    Returns:
        actions: [T, nu] ctrl array.
    """
    T = int(duration / control_dt)
    t = np.arange(T) * control_dt                          # [T]
    actions = amplitudes[None, :] * np.sin(
        2.0 * np.pi * freq * t[:, None] + phases[None, :]  # [T, nu]
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
    freq: float,
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
    amplitude_max: float = 0.2,
    perturbation: float = 0.2,
    n_substeps: int = 10,
    w_q: float = 1.0,
    w_qdot: float = 1.0,
    popsize: int = 16,
    max_generations: int = 50,
    seed: int = 42,
) -> tuple[np.ndarray, dict, dict]:
    """
    CMA-ES over (freq, amplitudes[nu], phases[nu]) to maximise FIM
    trace for the target parameter groups.

    Optimization space (d = 1 + 2*nu, all normalized to [-1, 1]):
        x[0]         → frequency   (log-uniform in freq_range)
        x[1:nu+1]    → amplitudes  (uniform in [0, amplitude_max])
        x[nu+1:2nu+1]→ phases      (uniform in [0, 2π])

    Initialized near the cheetah baseline (freq=2Hz, uniform amp,
    evenly-spaced phases) so optimization starts from a known-good point.

    Returns:
        best_actions: [T, nu] optimal ctrl sequence.
        best_params:  dict with freq, amplitudes, phases.
        history:      dict with per-generation FIM traces.
    """
    from cmaes import CMA

    nu = mj_model.nu
    d_opt = 1 + 2 * nu
    log_fmin = np.log(freq_range[0])
    log_fmax = np.log(freq_range[1])

    target_params = [p.name for p in param_space.params if p.group in target_groups]

    def decode(x: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        freq = float(np.exp(
            log_fmin + (np.clip(x[0], -1, 1) + 1) / 2.0 * (log_fmax - log_fmin)
        ))
        amplitudes = amplitude_max * (np.clip(x[1:nu+1], -1, 1) + 1) / 2.0
        phases = np.pi * (np.clip(x[nu+1:], -1, 1) + 1)   # [0, 2π]
        return freq, amplitudes, phases

    # Initialise near cheetah baseline
    x0_freq  = 2.0 * (np.log(2.0) - log_fmin) / (log_fmax - log_fmin) - 1.0
    x0_amp   = np.zeros(nu)                                    # → amplitude_max/2
    x0_phase = np.linspace(-1.0, 1.0, nu, endpoint=False)     # evenly spaced
    x0 = np.concatenate([[x0_freq], x0_amp, x0_phase])

    bounds = np.array([[-1.0, 1.0]] * d_opt)
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

    print(f"\nOptimising sinusoidal excitation (d={d_opt}: 1 freq + {nu} amp + {nu} phase)")
    print(f"  target groups: {target_groups}")
    print(f"  popsize={popsize}, max_gen={max_generations}, "
          f"duration={duration}s, A_max={amplitude_max}rad\n")

    for gen in range(max_generations):
        solutions = [optimizer.ask() for _ in range(popsize)]
        traces    = []

        for x in solutions:
            freq, amplitudes, phases = decode(x)
            try:
                fim, _ = compute_sinusoidal_fim(
                    mj_model, param_space,
                    freq, amplitudes, phases,
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

        if gen_best > best_trace:
            best_trace = gen_best
            best_x     = solutions[int(np.argmax(traces))]
            freq, amplitudes, phases = decode(best_x)
            best_params = {
                "freq":       float(freq),
                "amplitudes": amplitudes.tolist(),
                "phases":     phases.tolist(),
                "duration":   duration,
                "control_dt": control_dt,
            }
            _, best_actions = compute_sinusoidal_fim(
                mj_model, param_space,
                freq, amplitudes, phases,
                duration, control_dt, perturbation, n_substeps,
                w_q, w_qdot,
            )

        print(f"  Gen {gen:3d}: best={gen_best:.4f}  mean={gen_mean:.4f}  "
              f"freq={decode(solutions[int(np.argmax(traces))])[0]:.2f}Hz")

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

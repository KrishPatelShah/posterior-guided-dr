"""
Visualize the T1 environment: run a short rollout and plot trajectories.

Loads the T1 model from MuJoCo Playground, wraps it in PGDREnv with
uniform domain randomization (C1), rolls out with a sinusoidal action
pattern, and saves trajectory plots to pgdr/test/figures/.

Usage:
    python pgdr/test/visualize_t1.py [--steps 200] [--num-envs 1] [--seed 0]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Imports from project
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pgdr.t1_env import PGDREnv, OBS_DIM, ACT_DIM
from pgdr.param_space import build_t1_param_space
from pgdr.pgdr_randomizer import PGDRRandomizer, DRMode


# ---------------------------------------------------------------------------
# Load T1 model
# ---------------------------------------------------------------------------

def load_t1_model():
    """Load T1 MjModel from MuJoCo Playground registry."""
    from mujoco_playground import registry
    env = registry.load("T1JoystickFlatTerrain")
    return env.mj_model


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def run_rollout(
    env: PGDREnv,
    num_steps: int,
    num_envs: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """
    Roll out the environment with a sinusoidal action pattern.

    Returns a dict of trajectory arrays, each shaped [num_steps, num_envs, ...].
    """
    rng = jax.random.PRNGKey(seed)

    # JIT-compile step
    jit_step = jax.jit(env.step)

    # Reset
    state = env.reset(rng, num_envs=num_envs)

    # Storage (only first env for clarity)
    traj = {
        "base_height":   [],
        "base_vx":       [],
        "base_vy":       [],
        "base_vz":       [],
        "cmd_vx":        [],
        "cmd_vy":        [],
        "cmd_wz":        [],
        "reward":        [],
        "joint_pos":     [],   # all 23 joints
        "joint_vel":     [],
        "done":          [],
    }

    t_arr = np.arange(num_steps) * env.control_dt

    for t in range(num_steps):
        # Sinusoidal action: gentle leg oscillation
        phase = 2 * np.pi * t / num_steps
        action = np.zeros((num_envs, ACT_DIM), dtype=np.float32)
        # Alternate hip and knee joints with a slow sine wave
        action[:, 0::3] = 0.2 * np.sin(phase)      # hip pitch joints
        action[:, 1::3] = 0.1 * np.sin(phase + 0.5) # knee joints
        action_jax = jnp.array(action)

        state = jit_step(state, action_jax)

        # Collect from env 0
        d = state.mjx_data
        traj["base_height"].append(float(d.qpos[0, 2]))
        traj["base_vx"].append(float(d.qvel[0, 0]))
        traj["base_vy"].append(float(d.qvel[0, 1]))
        traj["base_vz"].append(float(d.qvel[0, 2]))
        traj["cmd_vx"].append(float(state.command[0, 0]))
        traj["cmd_vy"].append(float(state.command[0, 1]))
        traj["cmd_wz"].append(float(state.command[0, 2]))
        traj["reward"].append(float(state.reward[0]))
        traj["joint_pos"].append(np.array(d.qpos[0, 7:]))   # [23]
        traj["joint_vel"].append(np.array(d.qvel[0, 6:]))   # [23]
        traj["done"].append(bool(state.done[0]))

    return {k: np.array(v) for k, v in traj.items()}, t_arr


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

FIGURE_DIR = Path(__file__).parent / "figures"


def plot_base_state(traj: dict, t: np.ndarray, out_dir: Path) -> None:
    """Plot base height and velocity vs. commanded velocity."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("T1 Base State", fontsize=14)

    # Base height
    ax = axes[0, 0]
    ax.plot(t, traj["base_height"], color="steelblue", linewidth=1.5)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="Fall threshold")
    ax.set_ylabel("Height (m)")
    ax.set_title("Base Height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # vx tracking
    ax = axes[0, 1]
    ax.plot(t, traj["base_vx"], color="steelblue", linewidth=1.5, label="actual vx")
    ax.plot(t, traj["cmd_vx"], color="orange", linestyle="--", linewidth=1.5, label="cmd vx")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Forward Velocity Tracking")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # vy tracking
    ax = axes[1, 0]
    ax.plot(t, traj["base_vy"], color="steelblue", linewidth=1.5, label="actual vy")
    ax.plot(t, traj["cmd_vy"], color="orange", linestyle="--", linewidth=1.5, label="cmd vy")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Lateral Velocity Tracking")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Reward
    ax = axes[1, 1]
    ax.plot(t, traj["reward"], color="green", linewidth=1.5)
    ax.set_ylabel("Reward")
    ax.set_xlabel("Time (s)")
    ax.set_title("Step Reward")
    ax.grid(True, alpha=0.3)

    # Mark fall events
    for axi in axes.flat:
        for i, d in enumerate(traj["done"]):
            if d:
                axi.axvline(t[i], color="red", alpha=0.3, linewidth=0.8)

    plt.tight_layout()
    path = out_dir / "base_state.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_joint_positions(traj: dict, t: np.ndarray, out_dir: Path) -> None:
    """Plot joint positions for all 23 joints."""
    joint_pos = traj["joint_pos"]  # [T, 23]
    n_joints = joint_pos.shape[1]

    # 3 columns, ceil(23/3) rows
    n_cols = 3
    n_rows = (n_joints + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.2))
    fig.suptitle("T1 Joint Positions", fontsize=14)
    axes = axes.flat

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(t, joint_pos[:, j], linewidth=1.2)
        ax.set_title(f"Joint {j}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for j in range(n_joints, n_rows * n_cols):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = out_dir / "joint_positions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_joint_velocities(traj: dict, t: np.ndarray, out_dir: Path) -> None:
    """Plot joint velocities for all 23 joints."""
    joint_vel = traj["joint_vel"]  # [T, 23]
    n_joints = joint_vel.shape[1]

    n_cols = 3
    n_rows = (n_joints + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.2))
    fig.suptitle("T1 Joint Velocities", fontsize=14)
    axes = axes.flat

    for j in range(n_joints):
        ax = axes[j]
        ax.plot(t, joint_vel[:, j], linewidth=1.2, color="darkorange")
        ax.set_title(f"Joint {j}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)

    for j in range(n_joints, n_rows * n_cols):
        axes[j].set_visible(False)

    plt.tight_layout()
    path = out_dir / "joint_velocities.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_observation_heatmap(traj: dict, t: np.ndarray, out_dir: Path) -> None:
    """
    Summarize the 58-dim observation vector over time as a heatmap.
    Shows structure: lin_vel, ang_vel, gravity, joint_pos, joint_vel, command.
    """
    # Reconstruct observation from trajectory components (58-dim layout)
    # obs = [lin_vel(3), ang_vel(3), gravity(3), joint_pos(23), joint_vel(23), cmd(3)]
    T = len(t)
    obs = np.zeros((T, OBS_DIM))
    obs[:, 0] = traj["base_vx"]
    obs[:, 1] = traj["base_vy"]
    obs[:, 2] = traj["base_vz"]
    obs[:, 9:32] = traj["joint_pos"]
    obs[:, 32:55] = traj["joint_vel"]
    obs[:, 55] = traj["cmd_vx"]
    obs[:, 56] = traj["cmd_vy"]
    obs[:, 57] = traj["cmd_wz"]

    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(
        obs.T, aspect="auto", origin="lower",
        extent=[t[0], t[-1], 0, OBS_DIM],
        cmap="RdBu_r", vmin=-2, vmax=2,
    )
    fig.colorbar(im, ax=ax, label="Value (clipped ±2)")

    # Annotate observation groups
    boundaries = [0, 3, 6, 9, 32, 55, 58]
    labels = ["lin_vel", "ang_vel", "gravity", "joint_pos (23)", "joint_vel (23)", "command"]
    for i, (lo, hi, label) in enumerate(zip(boundaries[:-1], boundaries[1:], labels)):
        ax.axhline(lo, color="white", linewidth=0.5, alpha=0.5)
        ax.text(t[-1] * 1.001, (lo + hi) / 2, label, fontsize=7,
                va="center", ha="left", color="black")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Observation dimension")
    ax.set_title("T1 Observation Vector Over Time")
    plt.tight_layout()
    path = out_dir / "observation_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Rendered frames (optional, uses mujoco viewer offscreen)
# ---------------------------------------------------------------------------

def render_frames(env: PGDREnv, num_frames: int, seed: int, out_dir: Path) -> None:
    """Render a few frames using MuJoCo's offscreen renderer and save as a grid."""
    try:
        import mujoco
        renderer = mujoco.Renderer(env.mj_model, height=240, width=320)
    except Exception as e:
        print(f"  Skipping render (renderer unavailable): {e}")
        return

    rng = jax.random.PRNGKey(seed + 99)
    state = env.reset(rng, num_envs=1)
    jit_step = jax.jit(env.step)

    frames = []
    total_steps = num_frames * 5  # render every 5 steps
    for i in range(total_steps):
        action = jnp.zeros((1, ACT_DIM))
        state = jit_step(state, action)
        if i % 5 == 0:
            # Copy physics state back to mj_data for rendering
            mj_data = mujoco.MjData(env.mj_model)
            mj_data.qpos[:] = np.array(state.mjx_data.qpos[0])
            mj_data.qvel[:] = np.array(state.mjx_data.qvel[0])
            mujoco.mj_forward(env.mj_model, mj_data)
            renderer.update_scene(mj_data, camera="track")
            frames.append(renderer.render().copy())

    if not frames:
        return

    n_cols = min(4, len(frames))
    n_rows = (len(frames) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3))
    fig.suptitle("T1 Rendered Frames (standing still)", fontsize=12)
    for i, frame in enumerate(frames):
        ax = axes.flat[i] if n_rows > 1 else axes[i] if n_cols > 1 else axes
        ax.imshow(frame)
        ax.axis("off")
        ax.set_title(f"t={i*5*env.control_dt:.2f}s", fontsize=8)
    for i in range(len(frames), n_rows * n_cols):
        axes.flat[i].set_visible(False)

    plt.tight_layout()
    path = out_dir / "rendered_frames.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize T1 environment rollout")
    parser.add_argument("--steps", type=int, default=300,
                        help="Number of control steps to simulate (default: 300)")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--no-render", action="store_true",
                        help="Skip rendered frame visualization")
    args = parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print("Loading T1 model from MuJoCo Playground...")
    mj_model = load_t1_model()
    print(f"  nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}, nbody={mj_model.nbody}")

    # --- Build param space and randomizer (C1: uniform DR, no sysid needed) ---
    print("Building parameter space...")
    param_space = build_t1_param_space(mj_model)
    print(f"  d={param_space.d}  "
          f"(friction={len(param_space.group_indices('friction'))}, "
          f"mass={len(param_space.group_indices('mass'))}, "
          f"actuator={len(param_space.group_indices('actuator'))}, "
          f"contact={len(param_space.group_indices('contact'))})")

    randomizer = PGDRRandomizer(
        param_space=param_space,
        mode=DRMode.UNIFORM,   # C1: no sysid results needed
    )
    print(f"  Randomizer mode: {randomizer.mode}")

    # --- Build env ---
    print("Creating PGDREnv...")
    env = PGDREnv(mj_model=mj_model, randomizer=randomizer)
    print(f"  obs_dim={env.obs_dim}, act_dim={env.act_dim}, "
          f"control_dt={env.control_dt}s, n_substeps={env.n_substeps}")

    # --- Rollout ---
    print(f"\nRunning rollout: {args.steps} steps, {args.num_envs} env(s), seed={args.seed}...")
    traj, t_arr = run_rollout(env, args.steps, args.num_envs, args.seed)

    # Print summary stats
    print(f"  Base height: min={traj['base_height'].min():.3f}  "
          f"max={traj['base_height'].max():.3f}  "
          f"mean={traj['base_height'].mean():.3f}")
    print(f"  Reward:      min={traj['reward'].min():.3f}  "
          f"max={traj['reward'].max():.3f}  "
          f"mean={traj['reward'].mean():.3f}")
    n_falls = int(np.sum(traj["done"]))
    print(f"  Falls (done): {n_falls}")

    # --- Plots ---
    print(f"\nSaving figures to {FIGURE_DIR}/")
    plot_base_state(traj, t_arr, FIGURE_DIR)
    plot_joint_positions(traj, t_arr, FIGURE_DIR)
    plot_joint_velocities(traj, t_arr, FIGURE_DIR)
    plot_observation_heatmap(traj, t_arr, FIGURE_DIR)

    if not args.no_render:
        print("Rendering frames...")
        render_frames(env, num_frames=8, seed=args.seed, out_dir=FIGURE_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()

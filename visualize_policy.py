"""
visualize_policy.py — Roll out a trained PPO checkpoint and save trajectory plots.

Usage:
    python visualize_policy.py --checkpoint pgdr/checkpoints/C4_pgdr_1.0_seed0/final.pkl
    python visualize_policy.py --checkpoint pgdr/checkpoints/C2_pure_sysid_seed0/final.pkl
    python visualize_policy.py \
        --checkpoint pgdr/checkpoints/C4_pgdr_1.0_seed0/final.pkl \
        --results-dir pgdr/results \
        --steps 500 --out-dir pgdr/figures/rollout
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model():
    from pgdr.model_utils import load_mj_model
    return load_mj_model("t1")


def build_env(mj_model, p_star=None, results_dir=None):
    from pgdr.param_space import build_t1_param_space, ParamSpace
    from pgdr.pgdr_randomizer import PGDRRandomizer, DRMode
    from pgdr.t1_env import PGDREnv

    # Use param_space.json from results dir if available (matches p_star dims)
    ps = None
    if results_dir is not None:
        ps_path = Path(results_dir) / "param_space.json"
        if ps_path.exists():
            ps = ParamSpace.load(str(ps_path))
            print(f"  Loaded param_space from {ps_path} (d={ps.d})")

    if ps is None:
        ps = build_t1_param_space(mj_model)
        print(f"  Built full param_space (d={ps.d})")

    if p_star is not None:
        import copy
        mj_model_patched = copy.deepcopy(mj_model)
        ps.inject_cpu(mj_model_patched, np.array(p_star))
        print(f"  Injected p_star into model")
    else:
        mj_model_patched = mj_model

    p_star_jnp = jnp.array(p_star) if p_star is not None else jnp.zeros(ps.d)
    randomizer = PGDRRandomizer(param_space=ps, mode=DRMode.NONE, p_star=p_star_jnp)
    env = PGDREnv(mj_model=mj_model_patched, randomizer=randomizer)
    return env


def run_rollout(env, policy_fn, commands, control_dt, seed=0):
    """
    Run a single episode following a command sequence.

    commands: list of {vx, vy, wz, duration}
    Returns dict of trajectory arrays.
    """
    import mujoco
    from mujoco import mjx
    from pgdr.t1_env import OBS_DIM, ACT_DIM, _build_obs

    rng = jax.random.PRNGKey(seed)
    jit_step = jax.jit(env.step)

    # Build full command sequence as array
    cmd_steps = []
    for seg in commands:
        n = int(seg["duration"] / control_dt)
        cmd_steps.extend([(seg["vx"], seg["vy"], seg["wz"])] * n)
    total_steps = len(cmd_steps)
    cmd_array = jnp.array(cmd_steps, dtype=jnp.float32)  # [T, 3]

    # Reset with first command
    state = env.reset(rng, num_envs=1)

    traj = {
        "t":          [],
        "base_height": [],
        "base_vx":    [],
        "base_vy":    [],
        "base_wz":    [],
        "cmd_vx":     [],
        "cmd_vy":     [],
        "cmd_wz":     [],
        "reward":     [],
        "done":       [],
        "joint_pos":  [],
        "joint_vel":  [],
    }

    for i in range(total_steps):
        cmd = cmd_array[i]  # [3]
        # Override environment command
        state = state._replace(command=cmd[None])  # [1, 3]

        # Use stored obs from state (built with current command during step/reset)
        action = policy_fn(state.obs[0])  # [act_dim]
        state = jit_step(state, action[None])  # env expects [1, act_dim]

        d = state.mjx_data
        traj["t"].append(i * control_dt)
        traj["base_height"].append(float(d.qpos[0, 2]))
        traj["base_vx"].append(float(d.qvel[0, 0]))
        traj["base_vy"].append(float(d.qvel[0, 1]))
        traj["base_wz"].append(float(d.qvel[0, 5]))
        traj["cmd_vx"].append(float(cmd[0]))
        traj["cmd_vy"].append(float(cmd[1]))
        traj["cmd_wz"].append(float(cmd[2]))
        traj["reward"].append(float(state.reward[0]))
        traj["done"].append(bool(state.done[0]))
        traj["joint_pos"].append(np.array(d.qpos[0, 7:]))
        traj["joint_vel"].append(np.array(d.qvel[0, 6:]))

        if state.done[0]:
            print(f"  Episode ended at step {i} ({i*control_dt:.1f}s)")
            # pad remaining with last values
            for _ in range(i + 1, total_steps):
                for k in traj:
                    traj[k].append(traj[k][-1])
            break

    return {k: np.array(v) for k, v in traj.items()}


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_velocity_tracking(traj, out_dir, title_suffix=""):
    t = traj["t"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Velocity Tracking{title_suffix}", fontsize=13)

    ax = axes[0, 0]
    ax.plot(t, traj["base_height"], color="steelblue", lw=1.5)
    ax.axhline(0.3, color="red", ls="--", lw=1, label="fall threshold")
    ax.set_ylabel("Base height (m)")
    ax.set_title("Base Height")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, traj["base_vx"], color="steelblue", lw=1.5, label="actual")
    ax.plot(t, traj["cmd_vx"], color="orange", ls="--", lw=1.5, label="command")
    ax.set_ylabel("vx (m/s)")
    ax.set_title("Forward Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, traj["base_vy"], color="steelblue", lw=1.5, label="actual")
    ax.plot(t, traj["cmd_vy"], color="orange", ls="--", lw=1.5, label="command")
    ax.set_ylabel("vy (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Lateral Velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, traj["base_wz"], color="steelblue", lw=1.5, label="actual")
    ax.plot(t, traj["cmd_wz"], color="orange", ls="--", lw=1.5, label="command")
    ax.set_ylabel("wz (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Yaw Rate")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mark falls
    for ax in axes.flat:
        for i, done in enumerate(traj["done"]):
            if done:
                ax.axvline(t[i], color="red", alpha=0.4, lw=0.8)

    plt.tight_layout()
    path = out_dir / "velocity_tracking.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reward(traj, out_dir, title_suffix=""):
    t = traj["t"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, traj["reward"], color="green", lw=1.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Step reward")
    ax.set_title(f"Step Reward{title_suffix}")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = out_dir / "reward.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_joint_positions(traj, out_dir, title_suffix=""):
    joint_pos = traj["joint_pos"]  # [T, 23]
    t = traj["t"]
    n = joint_pos.shape[1]
    n_cols, n_rows = 4, (n + 3) // 4

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.2))
    fig.suptitle(f"Joint Positions{title_suffix}", fontsize=13)
    for j in range(n):
        ax = axes.flat[j]
        ax.plot(t, joint_pos[:, j], lw=1.0)
        ax.set_title(f"Joint {j}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
    for j in range(n, n_rows * n_cols):
        axes.flat[j].set_visible(False)

    plt.tight_layout()
    path = out_dir / "joint_positions.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def render_frames(env, policy_fn, commands, control_dt, out_dir, n_frames=12, seed=0):
    """Save a grid of rendered frames using MuJoCo offscreen renderer."""
    import mujoco
    try:
        renderer = mujoco.Renderer(env.mj_model, height=360, width=480)
    except Exception as e:
        print(f"  Offscreen renderer unavailable: {e}")
        return

    jit_step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = env.reset(rng, num_envs=1)

    cmd_steps = []
    for seg in commands:
        n = int(seg["duration"] / control_dt)
        cmd_steps.extend([(seg["vx"], seg["vy"], seg["wz"])] * n)
    cmd_array = jnp.array(cmd_steps, dtype=jnp.float32)

    total_steps = len(cmd_steps)
    render_every = max(1, total_steps // n_frames)
    frames, frame_times = [], []

    mj_data = mujoco.MjData(env.mj_model)

    for i in range(total_steps):
        cmd = cmd_array[i]
        state = state._replace(command=cmd[None])
        action = policy_fn(state.obs[0])
        state = jit_step(state, action[None])

        if i % render_every == 0:
            mj_data.qpos[:] = np.array(state.mjx_data.qpos[0])
            mj_data.qvel[:] = np.array(state.mjx_data.qvel[0])
            mujoco.mj_forward(env.mj_model, mj_data)
            renderer.update_scene(mj_data)
            frames.append(renderer.render().copy())
            frame_times.append(i * control_dt)

        if state.done[0]:
            break

    if not frames:
        return

    n_cols = min(4, len(frames))
    n_rows = (len(frames) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.2))
    fig.suptitle("Rendered Frames", fontsize=12)
    axes_flat = np.array(axes).flat
    for i, (frame, ft) in enumerate(zip(frames, frame_times)):
        ax = axes_flat[i]
        ax.imshow(frame)
        ax.set_title(f"t={ft:.1f}s", fontsize=8)
        ax.axis("off")
    for i in range(len(frames), n_rows * n_cols):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    path = out_dir / "rendered_frames.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualize trained PPO policy rollout")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to final.pkl checkpoint")
    parser.add_argument("--results-dir", default=None,
                        help="Sysid results dir for p_star.npy (optional, uses defaults if omitted)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory for figures (default: alongside checkpoint)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-render", action="store_true",
                        help="Skip offscreen rendered frames")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ckpt_path.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    condition = ckpt_path.parent.name
    title_suffix = f" — {condition}"

    # Load p_star if available
    p_star = None
    if args.results_dir:
        p_star_path = Path(args.results_dir) / "p_star.npy"
        if p_star_path.exists():
            p_star = np.load(p_star_path)
            print(f"Loaded p_star from {p_star_path}")

    print(f"Loading T1 model...")
    mj_model = load_model()

    print(f"Building environment...")
    env = build_env(mj_model, p_star=p_star, results_dir=args.results_dir)

    print(f"Loading policy from {ckpt_path}...")
    rng = jax.random.PRNGKey(args.seed)
    from pgdr._ppo import PPOAgent
    agent = PPOAgent.load(ckpt_path, rng)

    policy_fn = jax.jit(lambda obs: agent.get_deterministic_action(obs[None]).squeeze(0))

    # Command sequence
    commands = [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0, "duration": 5.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.5, "duration": 3.0},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": 3.0},
        {"vx": 0.0, "vy": 0.0, "wz": 0.0, "duration": 2.0},
    ]

    print(f"Running rollout ({sum(c['duration'] for c in commands):.0f}s)...")
    traj = run_rollout(env, policy_fn, commands, env.control_dt, seed=args.seed)

    n_falls = int(np.sum(np.diff(traj["done"].astype(int)) > 0))
    print(f"  Falls: {n_falls}")
    print(f"  Mean reward: {traj['reward'].mean():.4f}")
    print(f"  Mean vx error: {np.abs(traj['base_vx'] - traj['cmd_vx']).mean():.4f} m/s")

    print(f"\nSaving plots to {out_dir}/")
    plot_velocity_tracking(traj, out_dir, title_suffix)
    plot_reward(traj, out_dir, title_suffix)
    plot_joint_positions(traj, out_dir, title_suffix)

    if not args.no_render:
        print("Rendering frames...")
        render_frames(env, policy_fn, commands, env.control_dt, out_dir, seed=args.seed)

    print("\nDone.")


if __name__ == "__main__":
    main()

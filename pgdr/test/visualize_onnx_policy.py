"""
Visualize the ONNX policy across 5 friction parameter sets before running sysid.

Rolls out the policy under 5 different friction configurations and overlays the
resulting trajectories so you can confirm the policy is walking and that
different friction values produce distinguishable dynamics.

Usage:
    python pgdr/test/visualize_onnx_policy.py \\
        --onnx-policy path/to/policy.onnx \\
        --model-xml t1                      \\
        [--steps 500]                       \\
        [--perturbation 0.5]                \\
        [--seed 0]                          \\
        [--output-dir pgdr/test/figures]

The 5 trajectories use these friction configurations (in normalized space):
    0: default  — all zeros (nominal model)
    1: low      — all friction params at -perturbation
    2: high     — all friction params at +perturbation
    3: random A — random uniform in [-perturbation, +perturbation], seed
    4: random B — random uniform in [-perturbation, +perturbation], seed+1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco

from pgdr.model_utils import load_mj_model
from pgdr.param_space import build_t1_param_space
from pgdr.sysid import load_onnx_policy


# ---------------------------------------------------------------------------
# Rollout one trajectory under given friction params
# ---------------------------------------------------------------------------

def rollout_with_params(
    mj_model: mujoco.MjModel,
    param_space,
    p_normalized: np.ndarray,
    onnx_session,
    onnx_input_name: str,
    onnx_output_name: str,
    commands: list[dict],
    control_dt: float,
    n_substeps: int,
) -> dict[str, np.ndarray]:
    """
    Roll out the ONNX policy on a single mujoco world with the given params.

    Returns trajectory dict with arrays shaped [T].
    """
    # Save and inject params
    saved: dict[str, np.ndarray] = {}
    for param in param_space.params:
        key = param.mjx_field
        if key not in saved:
            saved[key] = getattr(mj_model, key).copy()
    param_space.inject_cpu(mj_model, p_normalized)

    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] = 0.78
        mj_data.qpos[3] = 1.0
    mujoco.mj_forward(mj_model, mj_data)

    # Policy state
    default_angles = (
        mj_model.key_qpos[0][7:].copy() if mj_model.nkey > 0
        else np.zeros(mj_model.nu)
    )
    last_action = np.zeros(mj_model.nu, dtype=np.float32)
    phase       = np.array([0.0, np.pi], dtype=np.float64)
    gait_freq   = 1.5
    phase_dt    = 2.0 * np.pi * gait_freq * control_dt

    base_height, base_vx, base_vy, cmd_vx_rec = [], [], [], []
    joint_pos_rec: list[np.ndarray] = []

    for cmd_dict in commands:
        command = np.array([
            cmd_dict.get("vx", 0.0),
            cmd_dict.get("vy", 0.0),
            cmd_dict.get("wz", 0.0),
        ], dtype=np.float32)
        n_steps = int(cmd_dict["duration"] / control_dt)

        for _ in range(n_steps):
            # Build 85-dim obs matching the playground T1 policy
            linvel   = mj_data.sensor("local_linvel").data.copy()
            gyro     = mj_data.sensor("gyro").data.copy()
            imu_id   = mj_model.site("imu").id
            imu_xmat = mj_data.site_xmat[imu_id].reshape(3, 3)
            gravity  = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
            jang     = mj_data.qpos[7:] - default_angles
            jvel     = mj_data.qvel[6:].copy()
            jang[:2] = 0.0
            jvel[:2] = 0.0
            ph       = phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
            phase4   = np.concatenate([np.cos(ph), np.sin(ph)])
            obs      = np.concatenate([linvel, gyro, gravity, command,
                                       jang, jvel, last_action, phase4]).astype(np.float32)

            onnx_pred = onnx_session.run(
                [onnx_output_name],
                {onnx_input_name: obs[None]},
            )[0][0]

            ctrl = onnx_pred + default_angles
            mj_data.ctrl[:] = ctrl
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            last_action = onnx_pred.copy()
            phase = np.fmod(phase + phase_dt + np.pi, 2.0 * np.pi) - np.pi

            base_height.append(mj_data.qpos[2])
            base_vx.append(mj_data.qvel[0])
            base_vy.append(mj_data.qvel[1])
            cmd_vx_rec.append(float(command[0]))
            joint_pos_rec.append(mj_data.qpos[7:].copy())

    # Restore model
    for key, val in saved.items():
        getattr(mj_model, key)[:] = val

    return {
        "base_height": np.array(base_height),
        "base_vx":     np.array(base_vx),
        "base_vy":     np.array(base_vy),
        "cmd_vx":      np.array(cmd_vx_rec),
        "joint_pos":   np.array(joint_pos_rec),  # [T, nu]
    }


# ---------------------------------------------------------------------------
# Build the 5 friction parameter sets
# ---------------------------------------------------------------------------

def make_friction_param_sets(
    n_friction: int,
    perturbation: float,
    seed: int,
) -> list[tuple[str, np.ndarray]]:
    """Return [(label, p_normalized), ...] for 5 friction configs."""
    rng = np.random.default_rng(seed)
    return [
        ("default (0)",    np.zeros(n_friction)),
        (f"low  (-{perturbation:.1f})",  np.full(n_friction, -perturbation)),
        (f"high (+{perturbation:.1f})",  np.full(n_friction, +perturbation)),
        ("random A",       rng.uniform(-perturbation, perturbation, n_friction)),
        ("random B",       rng.uniform(-perturbation, perturbation, n_friction)),
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple"]


def plot_trajectories(
    trajs: list[dict],
    labels: list[str],
    t: np.ndarray,
    out_dir: Path,
    n_substeps: int,
) -> None:
    """Save a multi-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("ONNX Policy — 5 Friction Parameter Sets", fontsize=13)

    # --- Base height ---
    ax = axes[0, 0]
    for i, (traj, label) in enumerate(zip(trajs, labels)):
        ax.plot(t, traj["base_height"], color=COLORS[i], linewidth=1.4,
                label=label, alpha=0.85)
    ax.axhline(0.3, color="red", linestyle="--", linewidth=1, label="fall threshold")
    ax.set_ylabel("Height (m)")
    ax.set_title("Base Height")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Forward velocity tracking ---
    ax = axes[0, 1]
    cmd_plotted = False
    for i, (traj, label) in enumerate(zip(trajs, labels)):
        ax.plot(t, traj["base_vx"], color=COLORS[i], linewidth=1.4,
                label=label, alpha=0.85)
        if not cmd_plotted:
            ax.plot(t, traj["cmd_vx"], color="black", linestyle="--",
                    linewidth=1.2, label="command vx", alpha=0.6)
            cmd_plotted = True
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Forward Velocity Tracking")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Hip pitch joints (joint indices for left/right hip pitch) ---
    ax = axes[1, 0]
    # Typical T1 joint ordering: indices differ per model; use first leg joints
    hip_idx = [0, 1, 2]   # first 3 joints (head/shoulder/hip-ish — adapt as needed)
    for i, (traj, label) in enumerate(zip(trajs, labels)):
        jp = traj["joint_pos"]
        if jp.shape[1] > 0:
            ax.plot(t, jp[:, hip_idx[0]], color=COLORS[i], linewidth=1.4,
                    label=label, alpha=0.85)
    ax.set_ylabel("Joint pos (rad)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Joint {hip_idx[0]} Position")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Knee joints ---
    ax = axes[1, 1]
    knee_candidates = [10, 11, 12, 13]
    knee_idx = next((j for j in knee_candidates
                     if trajs[0]["joint_pos"].shape[1] > j), 0)
    for i, (traj, label) in enumerate(zip(trajs, labels)):
        jp = traj["joint_pos"]
        ax.plot(t, jp[:, knee_idx], color=COLORS[i], linewidth=1.4,
                label=label, alpha=0.85)
    ax.set_ylabel("Joint pos (rad)")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Joint {knee_idx} Position")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = out_dir / "onnx_policy_friction_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")

    # --- Per-param joint position grid ---
    _plot_all_joints(trajs, labels, t, out_dir)


def _plot_all_joints(
    trajs: list[dict],
    labels: list[str],
    t: np.ndarray,
    out_dir: Path,
) -> None:
    """Plot all joints across trajectories in a grid."""
    n_joints = trajs[0]["joint_pos"].shape[1]
    n_cols = 4
    n_rows = (n_joints + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 2.0))
    fig.suptitle("ONNX Policy — All Joints, 5 Friction Configs", fontsize=11)
    ax_flat = axes.flat

    for j in range(n_joints):
        ax = ax_flat[j]
        for i, (traj, label) in enumerate(zip(trajs, labels)):
            ax.plot(t, traj["joint_pos"][:, j], color=COLORS[i],
                    linewidth=1.0, alpha=0.8, label=label if j == 0 else None)
        ax.set_title(f"Joint {j}", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.25)

    for j in range(n_joints, n_rows * n_cols):
        ax_flat[j].set_visible(False)

    # Single shared legend
    handles, lbls = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower right", fontsize=7, ncol=3)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    out = out_dir / "onnx_policy_all_joints.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Video recording
# ---------------------------------------------------------------------------

def record_video(
    mj_model: mujoco.MjModel,
    param_space,
    p_normalized: np.ndarray,
    onnx_session,
    onnx_input_name: str,
    onnx_output_name: str,
    commands: list[dict],
    control_dt: float,
    n_substeps: int,
    out_path: Path,
    fps: int = 50,
    height: int = 480,
    width: int = 640,
) -> None:
    """
    Roll out the ONNX policy with offscreen rendering and save as MP4.
    Requires imageio[ffmpeg]:  pip install imageio[ffmpeg]
    """
    try:
        import imageio
    except ImportError:
        print("  imageio not installed — skipping video. Run: pip install imageio[ffmpeg]")
        return

    print(f"\nRecording video → {out_path}")

    # Save and inject params
    saved: dict[str, np.ndarray] = {}
    for param in param_space.params:
        key = param.mjx_field
        if key not in saved:
            saved[key] = getattr(mj_model, key).copy()
    param_space.inject_cpu(mj_model, p_normalized)

    mj_data = mujoco.MjData(mj_model)
    if mj_model.nkey > 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    else:
        mujoco.mj_resetData(mj_model, mj_data)
        mj_data.qpos[2] = 0.78
        mj_data.qpos[3] = 1.0
    mujoco.mj_forward(mj_model, mj_data)

    default_angles = (
        mj_model.key_qpos[0][7:].copy() if mj_model.nkey > 0
        else np.zeros(mj_model.nu)
    )
    last_action = np.zeros(mj_model.nu, dtype=np.float32)
    phase       = np.array([0.0, np.pi], dtype=np.float64)
    gait_freq   = 1.5
    phase_dt    = 2.0 * np.pi * gait_freq * control_dt

    renderer = mujoco.Renderer(mj_model, height=height, width=width)
    frames: list[np.ndarray] = []

    for cmd_dict in commands:
        command = np.array([
            cmd_dict.get("vx", 0.0),
            cmd_dict.get("vy", 0.0),
            cmd_dict.get("wz", 0.0),
        ], dtype=np.float32)
        n_steps = int(cmd_dict["duration"] / control_dt)

        for step in range(n_steps):
            linvel   = mj_data.sensor("local_linvel").data.copy()
            gyro     = mj_data.sensor("gyro").data.copy()
            imu_id   = mj_model.site("imu").id
            imu_xmat = mj_data.site_xmat[imu_id].reshape(3, 3)
            gravity  = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
            jang     = mj_data.qpos[7:] - default_angles
            jvel     = mj_data.qvel[6:].copy()
            jang[:2] = 0.0
            jvel[:2] = 0.0
            ph       = phase if np.linalg.norm(command) >= 0.01 else np.ones(2) * np.pi
            phase4   = np.concatenate([np.cos(ph), np.sin(ph)])
            obs      = np.concatenate([linvel, gyro, gravity, command,
                                       jang, jvel, last_action, phase4]).astype(np.float32)

            onnx_pred = onnx_session.run(
                [onnx_output_name], {onnx_input_name: obs[None]}
            )[0][0]
            mj_data.ctrl[:] = onnx_pred + default_angles
            for _ in range(n_substeps):
                mujoco.mj_step(mj_model, mj_data)

            last_action = onnx_pred.copy()
            phase = np.fmod(phase + phase_dt + np.pi, 2.0 * np.pi) - np.pi

            renderer.update_scene(mj_data, camera="track")
            frames.append(renderer.render().copy())

            if step % 50 == 0:
                print(f"  {step}/{n_steps} steps rendered...", end="\r")

    renderer.close()

    # Restore model
    for key, val in saved.items():
        getattr(mj_model, key)[:] = val

    print(f"  Saving {len(frames)} frames at {fps} fps...")
    imageio.mimwrite(str(out_path), frames, fps=fps)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize ONNX policy across 5 friction parameter sets"
    )
    parser.add_argument("--onnx-policy", required=True, metavar="PATH",
                        help="Path to the ONNX policy file")
    parser.add_argument("--model-xml", default="t1",
                        help="Model XML path or 't1' alias (default: t1)")
    parser.add_argument("--steps", type=int, default=500,
                        help="Total control steps per trajectory (default: 500)")
    parser.add_argument("--perturbation", type=float, default=0.5,
                        help="Max friction perturbation in normalized space (default: 0.5)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for random friction sets (default: 0)")
    parser.add_argument("--control-dt", type=float, default=0.02,
                        help="Control timestep in seconds (default: 0.02)")
    parser.add_argument("--n-substeps", type=int, default=10,
                        help="Physics substeps per control step (default: 10)")
    parser.add_argument("--output-dir", default="pgdr/test/figures",
                        metavar="DIR")
    parser.add_argument("--record-video", action="store_true",
                        help="Record an MP4 of the default-friction rollout")
    parser.add_argument("--video-fps", type=int, default=50,
                        help="FPS for recorded video (default: 50)")
    parser.add_argument("--video-height", type=int, default=480)
    parser.add_argument("--video-width", type=int, default=640)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load model ---
    print(f"Loading model: {args.model_xml}")
    mj_model = load_mj_model(args.model_xml)
    print(f"  nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")

    # --- Friction-only param space ---
    full_ps = build_t1_param_space(mj_model)
    ps = full_ps.select_by_group(["friction"])
    print(f"Friction param space: d={ps.d}")

    # --- Load ONNX policy ---
    print(f"Loading ONNX policy: {args.onnx_policy}")
    session, input_name, output_name = load_onnx_policy(args.onnx_policy)

    # --- Command sequence covering args.steps total ---
    # Distribute steps across a forward walk + slight turn + backward walk
    total_duration = args.steps * args.control_dt
    seg = total_duration / 3.0
    commands = [
        {"vx": 1.0, "vy": 0.0, "wz": 0.0,  "duration": seg},
        {"vx": 0.5, "vy": 0.0, "wz": 0.3,  "duration": seg},
        {"vx": -0.5, "vy": 0.0, "wz": 0.0, "duration": seg},
    ]

    # --- Build 5 friction configs ---
    param_sets = make_friction_param_sets(ps.d, args.perturbation, args.seed)

    # --- Run 5 rollouts ---
    trajs = []
    labels = []
    total_steps_actual = sum(int(c["duration"] / args.control_dt) for c in commands)
    t = np.arange(total_steps_actual) * args.control_dt

    for idx, (label, p_norm) in enumerate(param_sets):
        print(f"\n[{idx+1}/5] Rolling out: {label}")
        traj = rollout_with_params(
            mj_model=mj_model,
            param_space=ps,
            p_normalized=p_norm,
            onnx_session=session,
            onnx_input_name=input_name,
            onnx_output_name=output_name,
            commands=commands,
            control_dt=args.control_dt,
            n_substeps=args.n_substeps,
        )
        trajs.append(traj)
        labels.append(label)

        n_steps = len(traj["base_height"])
        min_h   = traj["base_height"].min()
        mean_vx = traj["base_vx"].mean()
        print(f"  steps={n_steps}  min_height={min_h:.3f}  mean_vx={mean_vx:.3f}")
        if min_h < 0.3:
            print("  WARNING: robot fell (height < 0.3 m)")

    # --- Plot ---
    print(f"\nSaving figures to {out_dir}/")
    plot_trajectories(trajs, labels, t, out_dir, args.n_substeps)

    # --- Optional video recording (default-friction rollout only) ---
    if args.record_video:
        record_video(
            mj_model=mj_model,
            param_space=ps,
            p_normalized=param_sets[0][1],   # default friction
            onnx_session=session,
            onnx_input_name=input_name,
            onnx_output_name=output_name,
            commands=commands,
            control_dt=args.control_dt,
            n_substeps=args.n_substeps,
            out_path=out_dir / "rollout.mp4",
            fps=args.video_fps,
            height=args.video_height,
            width=args.video_width,
        )

    print("Done.")


if __name__ == "__main__":
    main()

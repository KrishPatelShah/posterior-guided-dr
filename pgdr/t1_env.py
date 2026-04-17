"""
MuJoCo Playground T1 joystick environment wrapper for PGDR.

Injects per-episode physical parameters sampled from the PGDR randomizer
into the MJX model at each episode reset.  Compatible with JAX jit and vmap.

Usage:
    env = PGDREnv(randomizer, model_xml)
    state = env.reset(rng)               # samples params, resets physics
    state, reward, done, info = env.step(state, action)

The wrapped env follows the MuJoCo Playground / Brax State interface:
    State.obs       [obs_dim]
    State.reward    scalar
    State.done      bool
    State.info      dict
    State.mjx_data  mjx.Data  (underlying physics state)
    State.mjx_model mjx.Model (current physics model, may be perturbed)
"""

from __future__ import annotations

from typing import Any, NamedTuple, Optional
import functools

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
import numpy as np


# ---------------------------------------------------------------------------
# State container (Brax-compatible NamedTuple)
# ---------------------------------------------------------------------------

class EnvState(NamedTuple):
    """Environment state compatible with JAX transforms."""
    mjx_data: mjx.Data
    mjx_model: mjx.Model     # current episode's physics params
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    step_count: jnp.ndarray
    command: jnp.ndarray     # [vx, vy, wz] velocity command
    rng: jax.Array           # per-env PRNG key for command resampling


CMD_RESAMPLE_INTERVAL = 100  # resample command every 2s (100 steps × 0.02s)


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _rotate_vec_by_quat_inv(quat: jnp.ndarray, vec: jnp.ndarray) -> jnp.ndarray:
    """Rotate vec from world frame to body frame using quaternion [w,x,y,z]."""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    # R^T (world->body) applied to vec
    vx, vy, vz = vec[0], vec[1], vec[2]
    rx = (1 - 2*(y*y + z*z))*vx + 2*(x*y + w*z)*vy  + 2*(x*z - w*y)*vz
    ry = 2*(x*y - w*z)*vx        + (1 - 2*(x*x + z*z))*vy + 2*(y*z + w*x)*vz
    rz = 2*(x*z + w*y)*vx        + 2*(y*z - w*x)*vy  + (1 - 2*(x*x + y*y))*vz
    return jnp.array([rx, ry, rz])


def _build_obs(
    data: mjx.Data,
    command: jnp.ndarray,
    model: mjx.Model,
    default_joint_pos: jnp.ndarray,
) -> jnp.ndarray:
    """
    Construct observation vector from MJX state.

    For the T1 joystick task, observation includes:
        [0:3]   base linear velocity (body frame)
        [3:6]   base angular velocity (body frame)
        [6:9]   gravity direction in body frame (points down when upright)
        [9:32]  joint positions relative to default (23 joints)
        [32:55] joint velocities (23 joints)
        [55:58] velocity command [vx, vy, wz]

    Total: 58-dim observation.
    """
    # Base state (freejoint: qpos[0:7], qvel[0:6])
    base_quat = data.qpos[3:7]            # quaternion [w, x, y, z]
    base_ang_vel = data.qvel[3:6]         # angular velocity (already body frame)
    world_lin_vel = data.qvel[0:3]        # linear velocity in world frame

    # Transform linear velocity to body frame
    base_lin_vel = _rotate_vec_by_quat_inv(base_quat, world_lin_vel)

    # Joint state relative to default pose
    joint_pos = data.qpos[7:] - default_joint_pos   # [23] relative to default
    joint_vel = data.qvel[6:]                         # [23]

    # Gravity vector in body frame: R^T @ [0,0,-1]
    # When upright, this gives [0, 0, -1] (gravity points down in body frame)
    gravity_world = jnp.array([0.0, 0.0, -1.0])
    gravity_body = _rotate_vec_by_quat_inv(base_quat, gravity_world)

    obs = jnp.concatenate([
        base_lin_vel,     # 3
        base_ang_vel,     # 3
        gravity_body,     # 3
        joint_pos,        # 23
        joint_vel,        # 23
        command,          # 3
    ])  # total: 58 dims
    return obs


OBS_DIM = 58   # matches _build_obs above
ACT_DIM = 23   # T1 has 23 actuated joints


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def _compute_reward(
    data: mjx.Data,
    prev_ctrl: jnp.ndarray,
    command: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Joystick velocity tracking reward for T1.

    Primary: track commanded linear + angular velocity.
    Penalties: joint velocity limits, action rate, orientation.
    """
    # Base velocity in body frame
    base_quat = data.qpos[3:7]
    world_lin_vel = data.qvel[0:3]
    base_lin_vel = _rotate_vec_by_quat_inv(base_quat, world_lin_vel)
    base_ang_vel = data.qvel[3:6]

    vx_cmd, vy_cmd, wz_cmd = command[0], command[1], command[2]

    # Linear velocity tracking (body frame vx/vy vs command)
    lin_vel_err = (base_lin_vel[0] - vx_cmd)**2 + (base_lin_vel[1] - vy_cmd)**2
    r_lin = jnp.exp(-4.0 * lin_vel_err)

    # Angular velocity tracking
    ang_vel_err = (base_ang_vel[2] - wz_cmd)**2
    r_ang = jnp.exp(-4.0 * ang_vel_err)

    # Upright orientation penalty
    w, x, y = base_quat[0], base_quat[1], base_quat[2]
    tilt = 2 * (x**2 + y**2)  # approx tilt from upright
    r_upright = jnp.exp(-5.0 * tilt)

    # Action smoothness penalty (prev_ctrl is ctrl from previous step)
    r_smooth = -0.01 * jnp.sum((action - prev_ctrl)**2)

    # Joint velocity penalty
    joint_vel = data.qvel[6:]
    r_jvel = -0.001 * jnp.sum(joint_vel**2)

    reward = 3.0 * r_lin + 0.5 * r_ang + 0.3 * r_upright + r_smooth + r_jvel
    return reward


# ---------------------------------------------------------------------------
# Done condition
# ---------------------------------------------------------------------------

def _check_done(data: mjx.Data, step_count: jnp.ndarray, max_steps: int) -> jnp.ndarray:
    """Check termination: base height too low, or max steps reached."""
    base_z = data.qpos[2]
    fallen = base_z < 0.3          # T1 base height threshold
    timeout = step_count >= max_steps
    return jnp.logical_or(fallen, timeout)


# ---------------------------------------------------------------------------
# Default T1 pose
# ---------------------------------------------------------------------------

def _get_default_qpos(mj_model: mujoco.MjModel) -> np.ndarray:
    """Return the default qpos from the model (keyframe 0 if present)."""
    if mj_model.nkey > 0:
        return mj_model.key_qpos[0].copy()
    qpos = np.zeros(mj_model.nq)
    qpos[2] = 0.78   # approximate standing height for T1
    qpos[3] = 1.0    # quaternion w=1 (upright)
    return qpos


# ---------------------------------------------------------------------------
# PGDREnv — main class
# ---------------------------------------------------------------------------

class PGDREnv:
    """
    T1 locomotion environment with PGDR parameter injection.

    At each episode reset, samples physical parameters from the randomizer
    and injects them into the MJX model, so different episodes train
    the policy under different dynamics.

    For C1 (uniform DR), uses the Playground default ranges.
    For C2-C4, randomizes according to the PGDR distribution.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        randomizer,                    # PGDRRandomizer instance
        control_dt: float = 0.02,      # 50 Hz
        sim_dt: float = 0.002,         # 500 Hz
        max_episode_steps: int = 1000,
        action_scale: float = 0.25,    # scale applied to policy actions
        command_ranges: Optional[dict] = None,
    ):
        self.mj_model = mj_model
        self.randomizer = randomizer
        self.control_dt = control_dt
        self.sim_dt = sim_dt
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.n_substeps = max(1, round(control_dt / sim_dt))
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM

        # Default command ranges [m/s or rad/s]
        self.command_ranges = command_ranges or {
            "vx": (-1.0, 1.0),
            "vy": (-0.5, 0.5),
            "wz": (-1.0, 1.0),
        }

        # Default joint positions (for qpos initialization and action offset)
        self._default_qpos = jnp.array(_get_default_qpos(mj_model))
        self._default_joint_pos = self._default_qpos[7:]  # [23]

        # Put the base model on device
        self._base_mjx_model = mjx.put_model(mj_model)

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array, num_envs: int = 1) -> EnvState:
        """
        Reset one or many environments with freshly sampled parameters.

        Args:
            rng:      PRNG key.
            num_envs: Number of parallel environments.

        Returns:
            EnvState with randomized physics models.
        """
        rng, param_rng, cmd_rng, init_rng = jax.random.split(rng, 4)

        # Sample physical parameters and inject into model copies
        mjx_model, _ = self.randomizer.apply_batch(
            param_rng, self._base_mjx_model, num_envs
        )

        # Sample velocity commands
        vx_lo, vx_hi = self.command_ranges["vx"]
        vy_lo, vy_hi = self.command_ranges["vy"]
        wz_lo, wz_hi = self.command_ranges["wz"]
        cmd_rngs = jax.random.split(cmd_rng, 3)
        commands = jnp.stack([
            jax.random.uniform(cmd_rngs[0], (num_envs,), minval=vx_lo, maxval=vx_hi),
            jax.random.uniform(cmd_rngs[1], (num_envs,), minval=vy_lo, maxval=vy_hi),
            jax.random.uniform(cmd_rngs[2], (num_envs,), minval=wz_lo, maxval=wz_hi),
        ], axis=-1)  # [num_envs, 3]

        # Initialize physics data from default qpos + small noise
        init_rngs = jax.random.split(init_rng, num_envs)
        # Per-env rngs for mid-episode command resampling
        step_rngs = jax.random.split(rng, num_envs)

        default_joint_pos = self._default_joint_pos

        def _init_single(mjx_m, cmd, rng_i):
            data = mjx.make_data(mjx_m)
            qpos = self._default_qpos + 0.01 * jax.random.normal(rng_i, (mjx_m.nq,))
            qpos = qpos.at[3:7].set(self._default_qpos[3:7])  # preserve quaternion
            data = data.replace(qpos=qpos)
            data = mjx.forward(mjx_m, data)
            obs = _build_obs(data, cmd, mjx_m, default_joint_pos)
            return data, obs

        # vmap over num_envs
        datas, obss = jax.vmap(_init_single)(mjx_model, commands, init_rngs)

        return EnvState(
            mjx_data=datas,
            mjx_model=mjx_model,
            obs=obss,
            reward=jnp.zeros(num_envs),
            done=jnp.zeros(num_envs, dtype=bool),
            step_count=jnp.zeros(num_envs, dtype=jnp.int32),
            command=commands,
            rng=step_rngs,
        )

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, state: EnvState, action: jnp.ndarray) -> EnvState:
        """
        Step environment(s) by one control timestep.

        Applies action, runs n_substeps of MJX physics, computes reward.
        Resets individual environments that are done (mid-batch auto-reset).

        Args:
            state:  Current EnvState (batched).
            action: [num_envs, act_dim] clipped actions.

        Returns:
            Next EnvState.
        """
        action = jnp.clip(action, -1.0, 1.0)
        n_substeps = self.n_substeps
        max_steps = self.max_episode_steps
        action_scale = self.action_scale
        default_joint_pos = self._default_joint_pos
        default_qpos = self._default_qpos
        vx_lo, vx_hi = self.command_ranges["vx"]
        vy_lo, vy_hi = self.command_ranges["vy"]
        wz_lo, wz_hi = self.command_ranges["wz"]

        def _step_single(mjx_m, data, cmd, act, step_count, rng):
            # Map normalized action to joint position targets around default pose
            motor_targets = default_joint_pos + action_scale * act
            # Record ctrl before stepping for smoothness penalty
            prev_ctrl = data.ctrl
            data = data.replace(ctrl=motor_targets)
            # Substep physics
            for _ in range(n_substeps):
                data = mjx.step(mjx_m, data)
            reward = _compute_reward(data, prev_ctrl, cmd, motor_targets, self.control_dt)
            new_step_count = step_count + 1
            done = _check_done(data, new_step_count, max_steps)
            # Auto-reset: if done, snap back to default pose so next rollout step
            # sees a valid state instead of a fallen/frozen robot.
            data = jax.tree_util.tree_map(
                lambda reset_val, live_val: jnp.where(done, reset_val, live_val),
                data.replace(
                    qpos=default_qpos,
                    qvel=jnp.zeros_like(data.qvel),
                    ctrl=jnp.zeros_like(data.ctrl),
                ),
                data,
            )
            step_count_out = jnp.where(done, jnp.zeros_like(new_step_count), new_step_count)
            # Resample command every CMD_RESAMPLE_INTERVAL steps or on done
            rng, cmd_rng = jax.random.split(rng)
            should_resample = jnp.logical_or(
                done, (new_step_count % CMD_RESAMPLE_INTERVAL) == 0
            )
            cmd_keys = jax.random.split(cmd_rng, 3)
            new_cmd = jnp.array([
                jax.random.uniform(cmd_keys[0], minval=vx_lo, maxval=vx_hi),
                jax.random.uniform(cmd_keys[1], minval=vy_lo, maxval=vy_hi),
                jax.random.uniform(cmd_keys[2], minval=wz_lo, maxval=wz_hi),
            ])
            cmd_out = jnp.where(should_resample, new_cmd, cmd)
            obs = _build_obs(data, cmd_out, mjx_m, default_joint_pos)
            return data, obs, reward, done, step_count_out, cmd_out, rng

        datas, obss, rewards, dones, step_counts, commands, rngs = jax.vmap(_step_single)(
            state.mjx_model, state.mjx_data, state.command,
            action, state.step_count, state.rng,
        )

        return state._replace(
            mjx_data=datas,
            obs=obss,
            reward=rewards,
            done=dones,
            step_count=step_counts,
            command=commands,
            rng=rngs,
        )

    # ------------------------------------------------------------------
    # Try loading from MuJoCo Playground (optional)
    # ------------------------------------------------------------------

    @classmethod
    def from_playground(
        cls,
        randomizer,
        task_name: str = "T1JoystickFlatTerrain",
        **kwargs,
    ) -> "PGDREnv":
        """
        Attempt to load the T1 model from MuJoCo Playground registry.

        Falls back to loading from the local XML if Playground is unavailable.
        """
        try:
            import mujoco_playground as mjp
            env = mjp.load(task_name)
            mj_model = env.mj_model
            print(f"Loaded T1 model from MuJoCo Playground: {task_name}")
        except ImportError:
            raise ImportError(
                "mujoco_playground not installed. Install with:\n"
                "  pip install mujoco-playground\n"
                "Or pass mj_model directly to PGDREnv()."
            )
        return cls(mj_model=mj_model, randomizer=randomizer, **kwargs)

    @classmethod
    def from_xml(
        cls,
        model_xml: str,
        randomizer,
        **kwargs,
    ) -> "PGDREnv":
        """Load from a local MuJoCo XML/MJCF file."""
        mj_model = mujoco.MjModel.from_xml_path(model_xml)
        return cls(mj_model=mj_model, randomizer=randomizer, **kwargs)

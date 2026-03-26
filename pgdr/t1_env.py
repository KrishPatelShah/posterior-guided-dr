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


# ---------------------------------------------------------------------------
# Observation builder
# ---------------------------------------------------------------------------

def _build_obs(data: mjx.Data, command: jnp.ndarray, model: mjx.Model) -> jnp.ndarray:
    """
    Construct observation vector from MJX state.

    For the T1 joystick task, observation includes:
        [0:3]   base linear velocity (world frame)
        [3:6]   base angular velocity (body frame)
        [6:10]  base quaternion (qpos[3:7])
        [10:33] joint positions relative to default (23 joints)
        [33:56] joint velocities (23 joints)
        [56:59] velocity command [vx, vy, wz]

    Total: 59-dim observation.
    """
    # Base state (freejoint: qpos[0:7], qvel[0:6])
    base_quat = data.qpos[3:7]            # quaternion [w, x, y, z]
    base_ang_vel = data.qvel[3:6]         # angular velocity
    base_lin_vel = data.qvel[0:3]         # linear velocity

    # Joint state (skip freejoint: first 7 qpos, first 6 qvel)
    joint_pos = data.qpos[7:]             # [23]
    joint_vel = data.qvel[6:]             # [23]

    # Gravity vector in body frame (projected from world z)
    # quat rotates world->body; gravity world = [0,0,-1]
    # Using simple approximation via quaternion rotation
    w, x, y, z = base_quat[0], base_quat[1], base_quat[2], base_quat[3]
    gravity_body = jnp.array([
        2*(x*z - w*y),
        2*(y*z + w*x),
        1 - 2*(x*x + y*y),
    ])  # body-frame projected gravity (unit vector pointing down)

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
    prev_data: mjx.Data,
    command: jnp.ndarray,
    action: jnp.ndarray,
    dt: float,
) -> jnp.ndarray:
    """
    Joystick velocity tracking reward for T1.

    Primary: track commanded linear + angular velocity.
    Penalties: joint velocity limits, action rate, orientation.
    """
    # Base velocity in world frame
    base_lin_vel = data.qvel[0:3]
    base_ang_vel = data.qvel[3:6]

    vx_cmd, vy_cmd, wz_cmd = command[0], command[1], command[2]

    # Linear velocity tracking
    lin_vel_err = (base_lin_vel[0] - vx_cmd)**2 + (base_lin_vel[1] - vy_cmd)**2
    r_lin = jnp.exp(-4.0 * lin_vel_err)

    # Angular velocity tracking
    ang_vel_err = (base_ang_vel[2] - wz_cmd)**2
    r_ang = jnp.exp(-4.0 * ang_vel_err)

    # Upright orientation penalty
    base_quat = data.qpos[3:7]
    w, x, y = base_quat[0], base_quat[1], base_quat[2]
    tilt = 2 * (x**2 + y**2)  # approx tilt from upright
    r_upright = jnp.exp(-5.0 * tilt)

    # Action smoothness penalty
    prev_action = prev_data.ctrl
    r_smooth = -0.01 * jnp.sum((action - prev_action)**2)

    # Joint velocity penalty
    joint_vel = data.qvel[6:]
    r_jvel = -0.001 * jnp.sum(joint_vel**2)

    reward = 1.5 * r_lin + 0.5 * r_ang + 0.3 * r_upright + r_smooth + r_jvel
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
        command_ranges: Optional[dict] = None,
    ):
        self.mj_model = mj_model
        self.randomizer = randomizer
        self.control_dt = control_dt
        self.sim_dt = sim_dt
        self.max_episode_steps = max_episode_steps
        self.n_substeps = max(1, round(control_dt / sim_dt))
        self.obs_dim = OBS_DIM
        self.act_dim = ACT_DIM

        # Default command ranges [m/s or rad/s]
        self.command_ranges = command_ranges or {
            "vx": (-1.0, 1.0),
            "vy": (-0.5, 0.5),
            "wz": (-1.0, 1.0),
        }

        # Default joint positions (for qpos initialization)
        self._default_qpos = jnp.array(_get_default_qpos(mj_model))

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

        def _init_single(mjx_m, cmd, rng_i):
            data = mjx.make_data(mjx_m)
            qpos = self._default_qpos + 0.01 * jax.random.normal(rng_i, (mjx_m.nq,))
            qpos = qpos.at[3:7].set(self._default_qpos[3:7])  # preserve quaternion
            data = data.replace(qpos=qpos)
            data = mjx.forward(mjx_m, data)
            obs = _build_obs(data, cmd, mjx_m)
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

        def _step_single(mjx_m, data, cmd, act, step_count):
            # Set control
            data = data.replace(ctrl=act)
            # Substep physics
            for _ in range(n_substeps):
                data = mjx.step(mjx_m, data)
            reward = _compute_reward(data, data, cmd, act, self.control_dt)
            step_count = step_count + 1
            done = _check_done(data, step_count, max_steps)
            obs = _build_obs(data, cmd, mjx_m)
            return data, obs, reward, done, step_count

        datas, obss, rewards, dones, step_counts = jax.vmap(_step_single)(
            state.mjx_model, state.mjx_data, state.command,
            action, state.step_count,
        )

        return state._replace(
            mjx_data=datas,
            obs=obss,
            reward=rewards,
            done=dones,
            step_count=step_counts,
        )

    # ------------------------------------------------------------------
    # Try loading from MuJoCo Playground (optional)
    # ------------------------------------------------------------------

    @classmethod
    def from_playground(
        cls,
        randomizer,
        task_name: str = "booster_t1-joystick",
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

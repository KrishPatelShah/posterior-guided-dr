"""
PPO agent in JAX/Flax for PGDR policy training on MuJoCo MJX.

Trains a locomotion policy for the Booster T1 humanoid with velocity tracking.
Domain randomization is applied externally by PGDRRandomizer before each episode.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
import mujoco
from mujoco import mjx


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    num_envs: int = 4096
    num_steps: int = 10
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    num_minibatches: int = 32
    update_epochs: int = 5
    hidden_dim: int = 256
    num_hidden: int = 2
    action_scale: float = 0.25
    sim_dt: float = 0.002
    control_dt: float = 0.02


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """MLP actor-critic for continuous control."""
    action_dim: int
    hidden_dim: int = 256
    num_hidden: int = 2

    @nn.compact
    def __call__(self, obs):
        x = obs
        for _ in range(self.num_hidden):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.elu(x)

        # Actor head
        action_mean = nn.Dense(self.action_dim)(x)
        action_log_std = self.param(
            "log_std",
            nn.initializers.constant(-0.5),
            (self.action_dim,),
        )

        # Critic head (separate last layer)
        vx = obs
        for _ in range(self.num_hidden):
            vx = nn.Dense(self.hidden_dim)(vx)
            vx = nn.elu(vx)
        value = nn.Dense(1)(vx)

        return action_mean, action_log_std, value.squeeze(-1)


# ---------------------------------------------------------------------------
# Observation & Reward
# ---------------------------------------------------------------------------

def make_obs(data: mjx.Data, cmd: jnp.ndarray) -> jnp.ndarray:
    """
    Build observation vector from MJX data + velocity command.

    Obs = [qpos (skip global x,y), qvel, cmd_vx, cmd_vy, cmd_wz]

    For a floating-base robot:
      qpos[0:3] = global position (x,y,z) — skip x,y to avoid drift
      qpos[2]   = height (keep)
      qpos[3:7] = quaternion orientation (keep)
      qpos[7:]  = joint angles (keep)
    """
    # Skip global x,y (indices 0,1) to make obs position-invariant
    qpos_obs = jnp.concatenate([data.qpos[2:7], data.qpos[7:]])
    return jnp.concatenate([qpos_obs, data.qvel, cmd])


def compute_reward(
    data: mjx.Data,
    prev_data: mjx.Data,
    action: jnp.ndarray,
    cmd: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Velocity tracking reward with regularization.

    cmd = [vx_target, vy_target, wz_target]

    Returns (reward, done).
    """
    # Base velocity (floating base: qvel[0:3] = linear, qvel[3:6] = angular)
    vx = data.qvel[0]
    vy = data.qvel[1]
    wz = data.qvel[5]

    # Velocity tracking (exponential kernel for smoother gradients)
    vel_err_sq = (vx - cmd[0])**2 + (vy - cmd[1])**2
    ang_err_sq = (wz - cmd[2])**2
    vel_reward = jnp.exp(-2.0 * vel_err_sq)
    ang_reward = jnp.exp(-2.0 * ang_err_sq)

    # Alive bonus (height-based)
    height = data.qpos[2]
    alive = jnp.where((height > 0.3) & (height < 1.5), 1.0, 0.0)

    # Regularization
    action_cost = 0.01 * jnp.sum(action**2)
    # Penalize joint velocities to encourage smooth motion
    joint_vel_cost = 0.001 * jnp.sum(data.qvel[6:]**2)
    # Penalize body angular velocity (roll/pitch) for stability
    orientation_cost = 0.1 * (data.qvel[3]**2 + data.qvel[4]**2)

    reward = (
        0.5 * vel_reward
        + 0.3 * ang_reward
        + 0.2 * alive
        - action_cost
        - joint_vel_cost
        - orientation_cost
    )

    # Terminate if robot falls
    done = jnp.where((height < 0.3) | (height > 2.0), 1.0, 0.0)

    return reward, done


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent for MJX batched environments."""

    def __init__(self, config: PPOConfig, obs_dim: int, act_dim: int, rng: jax.Array):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.network = ActorCritic(
            action_dim=act_dim,
            hidden_dim=config.hidden_dim,
            num_hidden=config.num_hidden,
        )

        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((obs_dim,))
        params = self.network.init(init_rng, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )

        self.state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
        )

        self.n_substeps = int(config.control_dt / config.sim_dt)

    @partial(jax.jit, static_argnums=(0,))
    def _get_action(self, params, obs, rng):
        """Sample action from policy (single env)."""
        action_mean, log_std, value = self.network.apply(params, obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, action_mean.shape)
        action = action_mean + std * noise
        log_prob = -0.5 * jnp.sum(
            ((action - action_mean) / std)**2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        return action, log_prob, value

    @partial(jax.jit, static_argnums=(0,))
    def _get_value(self, params, obs):
        _, _, value = self.network.apply(params, obs)
        return value

    def collect_rollout(
        self,
        batched_model: mjx.Model,
        batched_data: mjx.Data,
        cmd: jnp.ndarray,
        rng: jax.Array,
    ) -> dict:
        """
        Collect num_steps transitions from batched MJX environments.

        Args:
            batched_model: [num_envs, ...] batched MJX model.
            batched_data:  [num_envs, ...] batched MJX data (current state).
            cmd:           [num_envs, 3] velocity commands.
            rng:           PRNG key.

        Returns:
            dict with obs, actions, log_probs, values, rewards, dones, next_data.
        """
        num_steps = self.config.num_steps
        num_envs = self.config.num_envs
        params = self.state.params
        n_substeps = self.n_substeps
        action_scale = self.config.action_scale

        def env_step(carry, _):
            data, rng = carry
            rng, action_rng = jax.random.split(rng)

            # Observe
            obs = jax.vmap(make_obs)(data, cmd)  # [num_envs, obs_dim]

            # Act
            action_rngs = jax.random.split(action_rng, num_envs)
            actions, log_probs, values = jax.vmap(
                self._get_action, in_axes=(None, 0, 0)
            )(params, obs, action_rngs)

            # Scale actions and apply to ctrl
            ctrl = actions * action_scale
            prev_data = data
            data = data.replace(ctrl=ctrl)

            # Physics substeps
            def substep_fn(d, _):
                return jax.vmap(mjx.step)(batched_model, d), None
            data, _ = jax.lax.scan(substep_fn, data, None, length=n_substeps)

            # Reward
            rewards, dones = jax.vmap(compute_reward)(data, prev_data, actions, cmd)

            transition = {
                "obs": obs,
                "actions": actions,
                "log_probs": log_probs,
                "values": values,
                "rewards": rewards,
                "dones": dones,
            }

            return (data, rng), transition

        (next_data, _), rollout = jax.lax.scan(
            env_step, (batched_data, rng), None, length=num_steps
        )
        # rollout arrays have shape [num_steps, num_envs, ...]

        # Bootstrap value for GAE
        next_obs = jax.vmap(make_obs)(next_data, cmd)
        next_values = jax.vmap(self._get_value, in_axes=(None, 0))(params, next_obs)

        rollout["next_values"] = next_values
        rollout["next_data"] = next_data

        return rollout

    def update(self, rollout: dict) -> dict:
        """Run PPO update with GAE on collected rollout."""
        cfg = self.config

        # --- Compute GAE ---
        rewards = rollout["rewards"]      # [T, N]
        values = rollout["values"]        # [T, N]
        dones = rollout["dones"]          # [T, N]
        next_values = rollout["next_values"]  # [N]

        def compute_gae(rewards, values, dones, next_values):
            """Compute GAE advantages and returns."""
            T = rewards.shape[0]

            def gae_step(carry, t):
                gae = carry
                delta = rewards[t] + cfg.gamma * next_values * (1 - dones[t]) - values[t]
                # For intermediate steps, next_values should be values[t+1]
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * gae
                return gae, gae

            # Process in reverse
            rewards_rev = jnp.flip(rewards, axis=0)
            values_rev = jnp.flip(values, axis=0)
            dones_rev = jnp.flip(dones, axis=0)

            def gae_step_rev(gae, idx):
                # next_val for step t is values[t+1] or bootstrapped value
                next_val = jnp.where(
                    idx == 0,
                    next_values,
                    jnp.flip(values, axis=0)[idx - 1]
                )
                delta = rewards_rev[idx] + cfg.gamma * next_val * (1 - dones_rev[idx]) - values_rev[idx]
                gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones_rev[idx]) * gae
                return gae, gae

            _, advantages_rev = jax.lax.scan(
                gae_step_rev,
                jnp.zeros_like(next_values),
                jnp.arange(T),
            )
            advantages = jnp.flip(advantages_rev, axis=0)
            returns = advantages + values
            return advantages, returns

        advantages, returns = compute_gae(rewards, values, dones, next_values)

        # --- Flatten for minibatch updates ---
        # [T, N, ...] -> [T*N, ...]
        batch_size = cfg.num_steps * cfg.num_envs
        obs_flat = rollout["obs"].reshape(batch_size, -1)
        actions_flat = rollout["actions"].reshape(batch_size, -1)
        old_log_probs_flat = rollout["log_probs"].reshape(batch_size)
        advantages_flat = advantages.reshape(batch_size)
        returns_flat = returns.reshape(batch_size)

        # Normalize advantages
        advantages_flat = (advantages_flat - jnp.mean(advantages_flat)) / (jnp.std(advantages_flat) + 1e-8)

        # --- PPO epochs ---
        def ppo_loss_fn(params, obs, actions, old_log_probs, advantages, returns):
            action_mean, log_std, values = jax.vmap(
                lambda o: self.network.apply(params, o)
            )(obs)
            std = jnp.exp(log_std)

            # New log probs
            new_log_probs = -0.5 * jnp.sum(
                ((actions - action_mean) / std)**2 + 2 * log_std + jnp.log(2 * jnp.pi),
                axis=-1,
            )

            # Policy loss (clipped)
            ratio = jnp.exp(new_log_probs - old_log_probs)
            pg_loss1 = -advantages * ratio
            pg_loss2 = -advantages * jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps)
            policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

            # Value loss
            value_loss = 0.5 * jnp.mean((values - returns)**2)

            # Entropy bonus
            entropy = 0.5 * jnp.sum(log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e), axis=-1)
            entropy_loss = -jnp.mean(entropy)

            total_loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
            return total_loss, {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": -entropy_loss,
                "total_loss": total_loss,
            }

        @jax.jit
        def update_step(state, obs, actions, old_log_probs, advantages, returns):
            grad_fn = jax.value_and_grad(ppo_loss_fn, has_aux=True)
            (_, info), grads = grad_fn(state.params, obs, actions, old_log_probs, advantages, returns)
            state = state.apply_gradients(grads=grads)
            return state, info

        total_info = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        n_updates = 0
        minibatch_size = batch_size // cfg.num_minibatches

        rng = jax.random.PRNGKey(0)
        for _ in range(cfg.update_epochs):
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, batch_size)

            for mb in range(cfg.num_minibatches):
                idx = perm[mb * minibatch_size: (mb + 1) * minibatch_size]
                self.state, info = update_step(
                    self.state,
                    obs_flat[idx],
                    actions_flat[idx],
                    old_log_probs_flat[idx],
                    advantages_flat[idx],
                    returns_flat[idx],
                )
                for k in total_info:
                    total_info[k] += float(info[k])
                n_updates += 1

        # Average
        for k in total_info:
            total_info[k] /= max(n_updates, 1)

        mean_reward = float(jnp.mean(jnp.sum(rollout["rewards"], axis=0)))
        total_info["mean_episode_reward"] = mean_reward

        return total_info

    def save(self, path: str | Path):
        """Save agent state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "params": jax.device_get(self.state.params),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            }, f)

    def load(self, path: str | Path):
        """Load agent state from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=data["params"],
            tx=tx,
        )

    def make_policy_fn(self):
        """Return a callable policy for evaluation."""
        params = self.state.params
        network = self.network

        def policy_fn(obs, rng):
            action_mean, _, _ = network.apply(params, obs)
            return action_mean  # Deterministic at eval time

        return policy_fn

"""
Minimal PPO scaffold in JAX for PGDR policy training.

IMPORTANT: This is a structural scaffold showing the PPO interface that
the PGDR training loop expects. The collect_rollout() and update() methods
contain simplified logic that should be replaced with MuJoCo Playground's
full Brax-based PPO training infrastructure when integrating with the
Playground repository.

To integrate with MuJoCo Playground:
    1. Use brax.training.agents.ppo as the training backend
    2. Hook PGDRRandomizer.apply_batch() into the environment's reset()
    3. Use Playground's reward functions and observation pipeline
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state
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
        # Shared trunk
        x = obs
        for _ in range(self.num_hidden):
            x = nn.Dense(self.hidden_dim)(x)
            x = nn.elu(x)

        # Actor head: mean and log_std
        action_mean = nn.Dense(self.action_dim)(x)
        action_log_std = self.param(
            "log_std",
            nn.initializers.zeros,
            (self.action_dim,),
        )

        # Critic head
        value = nn.Dense(1)(x)

        return action_mean, action_log_std, value.squeeze(-1)


# ---------------------------------------------------------------------------
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent that interfaces with MJX batched environments."""

    def __init__(self, config: PPOConfig, obs_dim: int, act_dim: int, rng: jax.Array):
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Initialize network
        self.network = ActorCritic(
            action_dim=act_dim,
            hidden_dim=config.hidden_dim,
            num_hidden=config.num_hidden,
        )

        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, obs_dim))
        params = self.network.init(init_rng, dummy_obs)

        # Optimizer with gradient clipping
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.learning_rate),
        )

        self.state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
        )

    def get_action(self, params, obs, rng):
        """Sample action from the policy."""
        action_mean, log_std, value = self.network.apply(params, obs)
        std = jnp.exp(log_std)
        action = action_mean + std * jax.random.normal(rng, action_mean.shape)
        log_prob = -0.5 * jnp.sum(
            ((action - action_mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )
        return action, log_prob, value

    def get_value(self, params, obs):
        """Get value estimate."""
        _, _, value = self.network.apply(params, obs)
        return value

    def collect_rollout(
        self,
        batched_model: mjx.Model,
        rng: jax.Array,
        episode_length: int,
        num_steps: int,
    ) -> dict:
        """
        Collect a rollout of num_steps from the batched environment.

        Args:
            batched_model:  Batched MJX model [num_envs, ...].
            rng:            PRNG key.
            episode_length: Max episode length (for resets).
            num_steps:      Steps to collect before PPO update.

        Returns:
            Dictionary with observations, actions, rewards, etc.
        """
        num_envs = self.config.num_envs

        # Initialize data for all envs
        rng, reset_rng = jax.random.split(rng)

        # Create initial data by batching mjx.make_data across models
        # In practice, this would use MuJoCo Playground's env.reset()
        @jax.jit
        def collect_step(carry, _):
            data, rng, step_count = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)

            # Observe: concatenate qpos and qvel
            obs = jnp.concatenate([data.qpos, data.qvel], axis=-1)

            # Act
            action, log_prob, value = jax.vmap(
                lambda o, r: self.get_action(self.state.params, o[None], r),
                in_axes=(0, 0),
            )(obs, jax.random.split(action_rng, num_envs))
            action = action.squeeze(1)
            log_prob = log_prob.squeeze(1) if log_prob.ndim > 1 else log_prob
            value = value.squeeze(1) if value.ndim > 1 else value

            # Step physics
            data = data.replace(ctrl=action)
            data = jax.vmap(mjx.step)(batched_model, data)

            # Simple reward: negative velocity tracking error
            # In practice, this would use MuJoCo Playground's reward function
            reward = -jnp.sum(data.qvel[:, :3] ** 2, axis=-1)

            step_count = step_count + 1

            transition = {
                "obs": obs,
                "action": action,
                "log_prob": log_prob,
                "value": value,
                "reward": reward,
            }

            return (data, rng, step_count), transition

        # This is a simplified rollout collection.
        # In integration with MuJoCo Playground, you would use the
        # environment's step/reset/reward functions directly.

        return {"episode_returns": [0.0]}  # Placeholder

    def update(self, rollout_data: dict) -> dict:
        """
        Run PPO update on collected rollout data.

        Returns:
            Loss info dictionary.
        """
        # Placeholder — in integration with MuJoCo Playground,
        # this uses Brax's PPO update or a custom implementation.
        return {"total_loss": 0.0}

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
        # Reconstruct train state with loaded params
        tx = optax.chain(
            optax.clip_by_global_norm(self.config.max_grad_norm),
            optax.adam(self.config.learning_rate),
        )
        self.state = train_state.TrainState.create(
            apply_fn=self.network.apply,
            params=data["params"],
            tx=tx,
        )

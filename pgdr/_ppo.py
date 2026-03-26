"""
PPO implementation in JAX for PGDR policy training.

Designed to work with PGDREnv (t1_env.py).  Uses jax.lax.scan for
efficient rollout collection and supports JAX jit throughout.

Architecture: MLP actor-critic with separate actor/critic heads.
Training loop: GAE advantage estimation + clipped PPO objective.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flax.training import train_state


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PPOConfig:
    num_envs: int = 4096
    num_steps: int = 10           # rollout steps before update
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
    normalize_advantages: bool = True


# ---------------------------------------------------------------------------
# Transition container
# ---------------------------------------------------------------------------

class Transition(NamedTuple):
    obs: jnp.ndarray        # [T, num_envs, obs_dim]
    action: jnp.ndarray     # [T, num_envs, act_dim]
    log_prob: jnp.ndarray   # [T, num_envs]
    value: jnp.ndarray      # [T, num_envs]
    reward: jnp.ndarray     # [T, num_envs]
    done: jnp.ndarray       # [T, num_envs]


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class ActorCritic(nn.Module):
    """Separate-head MLP actor-critic for continuous control."""
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

        # Actor head
        action_mean = nn.Dense(self.action_dim)(x)
        action_log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(shape, -0.5),
            (self.action_dim,),
        )

        # Critic head
        value = nn.Dense(1)(x).squeeze(-1)

        return action_mean, action_log_std, value


# ---------------------------------------------------------------------------
# Gaussian log-probability
# ---------------------------------------------------------------------------

def gaussian_log_prob(action, mean, log_std):
    """Log prob of action under N(mean, exp(log_std)²)."""
    std = jnp.exp(log_std)
    return -0.5 * jnp.sum(
        ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1,
    )


def gaussian_entropy(log_std):
    """Entropy of diagonal Gaussian."""
    return 0.5 * jnp.sum(1.0 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)


# ---------------------------------------------------------------------------
# GAE computation
# ---------------------------------------------------------------------------

def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    last_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Compute Generalized Advantage Estimation.

    Args:
        rewards:    [T, num_envs]
        values:     [T, num_envs]
        dones:      [T, num_envs]
        last_value: [num_envs] — bootstrapped value after last step
        gamma:      discount factor
        gae_lambda: GAE lambda

    Returns:
        advantages: [T, num_envs]
        returns:    [T, num_envs]
    """
    T = rewards.shape[0]

    def _scan_fn(carry, t):
        next_adv, next_val = carry
        reward = rewards[T - 1 - t]
        value = values[T - 1 - t]
        done = dones[T - 1 - t]

        not_done = 1.0 - done.astype(jnp.float32)
        delta = reward + gamma * next_val * not_done - value
        adv = delta + gamma * gae_lambda * next_adv * not_done
        return (adv, value), adv

    _, advantages_reversed = jax.lax.scan(
        _scan_fn,
        (jnp.zeros_like(last_value), last_value),
        jnp.arange(T),
    )

    advantages = jnp.flip(advantages_reversed, axis=0)
    returns = advantages + values
    return advantages, returns


# ---------------------------------------------------------------------------
# PPO loss
# ---------------------------------------------------------------------------

def ppo_loss(
    params,
    network: ActorCritic,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
) -> tuple[jnp.ndarray, dict]:
    """PPO clipped objective."""
    action_mean, log_std, values = network.apply(params, obs)

    log_probs = gaussian_log_prob(actions, action_mean, log_std)
    entropy = gaussian_entropy(log_std)

    # Policy loss
    ratio = jnp.exp(log_probs - old_log_probs)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps)
    policy_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    # Value loss
    value_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
    entropy_loss = -jnp.mean(entropy)

    total_loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

    info = {
        "total_loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": -entropy_loss,
        "approx_kl": jnp.mean((ratio - 1) - jnp.log(ratio)),
    }
    return total_loss, info


# ---------------------------------------------------------------------------
# PPOAgent
# ---------------------------------------------------------------------------

class PPOAgent:
    """PPO agent that interfaces with PGDREnv."""

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
        dummy_obs = jnp.zeros((1, obs_dim))
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

        # JIT the inner functions
        self._jit_step = jax.jit(self._env_step)
        self._jit_update = jax.jit(self._update_epoch)

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def get_action_and_value(self, params, obs, rng):
        """Sample action, compute log_prob and value estimate."""
        action_mean, log_std, value = self.network.apply(params, obs)
        std = jnp.exp(log_std)
        noise = jax.random.normal(rng, action_mean.shape)
        action = action_mean + std * noise
        log_prob = gaussian_log_prob(action, action_mean, log_std)
        return action, log_prob, value

    def get_value(self, params, obs):
        _, _, value = self.network.apply(params, obs)
        return value

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _env_step(self, env, state, params, rng):
        """One environment step: get action, step env, return transition."""
        rng, act_rng = jax.random.split(rng)
        # Vectorize action sampling over num_envs
        act_rngs = jax.random.split(act_rng, state.obs.shape[0])
        actions, log_probs, values = jax.vmap(
            lambda o, r: self.get_action_and_value(params, o[None], r),
        )(state.obs, act_rngs)
        actions = actions.squeeze(1)
        log_probs = log_probs.squeeze(1) if log_probs.ndim > 1 else log_probs
        values = values.squeeze(1) if values.ndim > 1 else values

        # Step env
        next_state = env.step(state, actions)

        transition = Transition(
            obs=state.obs,
            action=actions,
            log_prob=log_probs,
            value=values,
            reward=next_state.reward,
            done=next_state.done,
        )
        return next_state, transition, rng

    def collect_rollout(self, env, env_state, num_steps: int, rng: jax.Array):
        """
        Collect num_steps of transitions from the environment.

        Returns:
            Transition namedtuple with leading dim = num_steps.
            next_env_state for bootstrapping.
        """
        transitions = []
        state = env_state

        for _ in range(num_steps):
            state, transition, rng = self._jit_step(env, state, self.state.params, rng)
            transitions.append(transition)

        # Stack into [T, num_envs, ...] tensors
        stacked = Transition(
            obs=jnp.stack([t.obs for t in transitions]),
            action=jnp.stack([t.action for t in transitions]),
            log_prob=jnp.stack([t.log_prob for t in transitions]),
            value=jnp.stack([t.value for t in transitions]),
            reward=jnp.stack([t.reward for t in transitions]),
            done=jnp.stack([t.done for t in transitions]),
        )
        return stacked, state, rng

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def _update_epoch(self, train_state_obj, obs_flat, actions_flat,
                      old_log_probs_flat, advantages_flat, returns_flat, rng):
        """One minibatch epoch of PPO updates."""
        n = obs_flat.shape[0]
        minibatch_size = n // self.config.num_minibatches

        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, n)

        total_info = {}

        def _mb_update(ts, mb_idx):
            start = mb_idx * minibatch_size
            idx = jax.lax.dynamic_slice(perm, (start,), (minibatch_size,))
            mb_obs = obs_flat[idx]
            mb_act = actions_flat[idx]
            mb_lp = old_log_probs_flat[idx]
            mb_adv = advantages_flat[idx]
            mb_ret = returns_flat[idx]

            grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
            (loss, info), grads = grad_fn(
                ts.params, self.network,
                mb_obs, mb_act, mb_lp, mb_adv, mb_ret,
                self.config.clip_eps,
                self.config.entropy_coef,
                self.config.value_coef,
            )
            ts = ts.apply_gradients(grads=grads)
            return ts, info

        for mb_idx in range(self.config.num_minibatches):
            train_state_obj, info = _mb_update(train_state_obj, mb_idx)
            total_info = info  # keep last minibatch info

        return train_state_obj, total_info, rng

    def update(
        self,
        rollout: Transition,
        last_value: jnp.ndarray,
        rng: jax.Array,
    ) -> tuple[dict, jax.Array]:
        """
        Run PPO update on a collected rollout.

        Args:
            rollout:    Transition with shape [T, num_envs, ...].
            last_value: [num_envs] bootstrapped value for final state.
            rng:        PRNG key.

        Returns:
            (loss_info dict, updated rng)
        """
        cfg = self.config

        # Compute advantages and returns
        advantages, returns = compute_gae(
            rollout.reward, rollout.value, rollout.done,
            last_value, cfg.gamma, cfg.gae_lambda,
        )

        # Normalize advantages
        if cfg.normalize_advantages:
            adv_mean = jnp.mean(advantages)
            adv_std = jnp.std(advantages) + 1e-8
            advantages = (advantages - adv_mean) / adv_std

        # Flatten time and env dims: [T * num_envs, ...]
        T, N = rollout.obs.shape[:2]
        flat = lambda x: x.reshape(T * N, *x.shape[2:])

        obs_flat = flat(rollout.obs)
        actions_flat = flat(rollout.action)
        log_probs_flat = flat(rollout.log_prob)
        advantages_flat = flat(advantages)
        returns_flat = flat(returns)

        # Multiple epochs of PPO updates
        all_info = {}
        for _ in range(cfg.update_epochs):
            self.state, info, rng = self._jit_update(
                self.state,
                obs_flat, actions_flat, log_probs_flat,
                advantages_flat, returns_flat,
                rng,
            )
            all_info = info

        return {k: float(v) for k, v in all_info.items()}, rng

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "params": jax.device_get(self.state.params),
                "config": self.config,
                "obs_dim": self.obs_dim,
                "act_dim": self.act_dim,
            }, f)

    @classmethod
    def load(cls, path: str | Path, rng: jax.Array) -> "PPOAgent":
        with open(path, "rb") as f:
            data = pickle.load(f)
        agent = cls(data["config"], data["obs_dim"], data["act_dim"], rng)
        tx = optax.chain(
            optax.clip_by_global_norm(data["config"].max_grad_norm),
            optax.adam(data["config"].learning_rate),
        )
        agent.state = train_state.TrainState.create(
            apply_fn=agent.network.apply,
            params=data["params"],
            tx=tx,
        )
        return agent

    def get_deterministic_action(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Deterministic action (mean) for evaluation."""
        action_mean, _, _ = self.network.apply(self.state.params, obs)
        return jnp.clip(action_mean, -1.0, 1.0)
